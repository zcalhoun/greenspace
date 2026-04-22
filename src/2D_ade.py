"""
Steady-state 2D anisotropic advection-dispersion equation (ADE) solver.

Solves  v·∇C − ∇·(D∇C) = S(x)  via multiplication in Fourier space:

    Ĝ(k) = 1 / (i k_L v + D_L k_L² + D_T k_T²)

where k_L, k_T are wavenumbers in the streamwise / cross-stream frame.
The solution is then C = IFFT( Ĝ · FFT(S) ).

Wrap-around (periodic domain) artifacts are suppressed by zero-padding the
source field before the FFT and cropping the result afterward.  A safe padding
is ceil(3 * D_L / (v * dx)) cells — approximately 3 longitudinal decay lengths.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnisotropicADEGreen(nn.Module):
    """
    Spectral Green's function convolution for the steady-state 2D anisotropic ADE.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions (number of cells in x and y).
    dx, dy : float
        Physical grid spacing.
    v : float
        Advection speed along the streamwise axis.
    D_L : float
        Longitudinal (streamwise) dispersion coefficient.
    D_T : float
        Transverse (cross-stream) dispersion coefficient.
    theta : float
        Angle (radians) of the streamwise axis from the +x axis.  Full
        [0, 2π) range is required since the advection direction breaks the
        π-periodicity of the dispersion tensor alone.
    padding : int or (int, int)
        Number of zero-padding cells added to each side of the domain before
        the FFT, then stripped from the output.  This is the primary control
        for wrap-around artifacts: the plume must decay to near-zero within
        the padded region.  A safe default is ceil(3 * D_L / (v * dx)) cells
        (≈ 3 longitudinal decay lengths).  Pass 0 to disable.
    dc_reg : float or None
        Small regularisation added to the real part of the denominator at all
        wavenumbers, keeping Ĝ(0,0) finite and the output non-negative.
        Physically corresponds to a weak uniform linear decay (sink) term.
        If None (default), set automatically to D_L * (2π / Lx)² based on
        the physical (unpadded) domain, so it is independent of padding.
    normalize : bool
        If True (default), the kernel is divided by its DC value so that the
        weights sum to 1 and the output is a proper weighted average of the
        input.  If False, the output is in physical concentration units.
    learnable : bool
        If True, expose v, D_L, D_T, theta as nn.Parameters (in log-space for
        the positive scalars) so they can be optimised by an autograd loop.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float = 1.0,
        dy: float = 1.0,
        v: float = 1.0,
        D_L: float = 1.0,
        D_T: float = 0.1,
        theta: float = 0.0,
        padding: "int | tuple[int, int]" = 0,
        dc_reg: "float | None" = None,
        normalize: bool = True,
        learnable: bool = False,
    ):
        super().__init__()

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.normalize = normalize

        # dc_reg anchored to the physical (unpadded) domain so it does not
        # shrink as padding grows.
        if dc_reg is None:
            Lx = nx * dx
            dc_reg = D_L * (2 * math.pi / Lx) ** 2
        self.dc_reg = dc_reg

        # Normalise padding to (pad_x, pad_y).
        if isinstance(padding, int):
            pad_x = pad_y = padding
        else:
            pad_x, pad_y = padding
        self.pad_x = pad_x
        self.pad_y = pad_y

        nx_pad = nx + 2 * pad_x
        ny_pad = ny + 2 * pad_y

        # Wavenumber grids on the padded domain.
        kx = torch.fft.fftfreq(nx_pad, d=dx) * 2 * math.pi
        ky = torch.fft.fftfreq(ny_pad, d=dy) * 2 * math.pi
        Kx, Ky = torch.meshgrid(kx, ky, indexing="ij")
        self.register_buffer("Kx", Kx)
        self.register_buffer("Ky", Ky)

        # Physical parameters in log-space (v, D_L, D_T) to enforce positivity.
        _log_v = torch.tensor(math.log(v), dtype=torch.float64)
        _log_D_L = torch.tensor(math.log(D_L), dtype=torch.float64)
        _log_D_T = torch.tensor(math.log(D_T), dtype=torch.float64)
        _theta = torch.tensor(theta, dtype=torch.float64)

        if learnable:
            self.log_v = nn.Parameter(_log_v)
            self.log_D_L = nn.Parameter(_log_D_L)
            self.log_D_T = nn.Parameter(_log_D_T)
            self.theta = nn.Parameter(_theta)
        else:
            self.register_buffer("log_v", _log_v)
            self.register_buffer("log_D_L", _log_D_L)
            self.register_buffer("log_D_T", _log_D_T)
            self.register_buffer("theta", _theta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _params(self):
        """Return physical parameters, ensuring positivity via exp."""
        return (
            torch.exp(self.log_v),
            torch.exp(self.log_D_L),
            torch.exp(self.log_D_T),
            self.theta,
        )

    # ------------------------------------------------------------------
    # Green's function in Fourier space
    # ------------------------------------------------------------------

    def green_function_fourier(self) -> torch.Tensor:
        """
        Build and return the adjoint Ĝ*(k) as a complex tensor of shape
        (nx_pad, ny_pad).

        Ĝ*(k) = 1 / (D_L k_L² + D_T k_T²  −  i v k_L  +  ε)

        The negated advective term gives the adjoint (reflected) kernel, so
        the plume in physical space extends downstream from each output
        location — the correct orientation for a source-to-influence filter.
        """
        v, D_L, D_T, theta = self._params()

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        K_L = self.Kx * cos_t + self.Ky * sin_t
        K_T = -self.Kx * sin_t + self.Ky * cos_t

        denom_real = D_L * K_L**2 + D_T * K_T**2 + self.dc_reg
        denom_imag = -v * K_L  # negated → adjoint

        # 1 / (a + ib) = (a − ib) / (a² + b²)
        denom_sq = denom_real**2 + denom_imag**2
        G_hat_real = denom_real / denom_sq
        G_hat_imag = -denom_imag / denom_sq

        G_hat = torch.complex(G_hat_real, G_hat_imag)

        if self.normalize:
            G_hat = G_hat / G_hat[0, 0].real

        return G_hat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Convolve source field S with the adjoint steady-state Green's function.

        Parameters
        ----------
        S : Tensor, shape (..., nx, ny)
            Source term (e.g. one-hot land-use map).  Batch and channel
            dimensions are supported; the last two axes are spatial.

        Returns
        -------
        C : Tensor, shape (..., nx, ny)
            Non-negative output field (weighted average of S if normalize=True).
        """
        if self.pad_x > 0 or self.pad_y > 0:
            S = F.pad(S, (self.pad_y, self.pad_y, self.pad_x, self.pad_x))

        G_hat = self.green_function_fourier()
        S_hat = torch.fft.fft2(S)
        C_pad = torch.fft.ifft2(S_hat * G_hat).real

        if self.pad_x > 0 or self.pad_y > 0:
            C_pad = C_pad[
                ...,
                self.pad_x : self.pad_x + self.nx,
                self.pad_y : self.pad_y + self.ny,
            ]

        return C_pad

    # ------------------------------------------------------------------
    # Highest Density Region
    # ------------------------------------------------------------------

    def highest_density_region(
        self, fraction: float = 0.95
    ) -> "tuple[torch.Tensor, float, float]":
        """
        Find the smallest spatial region containing `fraction` of the kernel mass.

        Pixels are selected greedily in descending order of kernel value — this
        is the provably optimal (minimum-area) solution for a given mass target.

        The kernel is evaluated on the padded domain so the full tail is visible,
        then fftshift-ed so the origin (source location) sits at the centre.

        Parameters
        ----------
        fraction : float
            Target fraction of total mass to capture (default 0.95).

        Returns
        -------
        mask : BoolTensor, shape (nx_pad, ny_pad)
            True for pixels inside the HDR.  The centre pixel corresponds to
            the source location; axes are in physical (x, y) order.
        area : float
            Physical area of the HDR in the same units as dx * dy.
        actual_fraction : float
            Fraction of mass actually captured (≥ fraction, due to ties).
        """
        with torch.no_grad():
            G_hat = self.green_function_fourier()
            # fftshift centres the kernel so the source is at the middle pixel.
            G = torch.fft.fftshift(torch.fft.ifft2(G_hat).real)

        # Guard against tiny numerical negatives before normalising.
        G = G.clamp(min=0.0)
        total = G.sum()
        if total == 0:
            raise ValueError(
                "Kernel has zero total mass — check dc_reg and parameters."
            )
        G = G / total

        # Sort pixels by density, accumulate mass, find cutoff.
        flat = G.flatten()
        sorted_vals, _ = flat.sort(descending=True)
        cumsum = sorted_vals.cumsum(0)

        # Number of pixels needed: first index where cumsum reaches fraction.
        n_pixels = int((cumsum < fraction).sum().item()) + 1
        threshold = sorted_vals[n_pixels - 1].item()

        mask = G >= threshold
        actual_fraction = G[mask].sum().item()
        area = mask.sum().item() * self.dx * self.dy

        return mask, area, actual_fraction

    # ------------------------------------------------------------------
    # Convenience: dispersion tensor in global Cartesian coordinates
    # ------------------------------------------------------------------

    def dispersion_tensor(self) -> torch.Tensor:
        """
        Return the 2×2 dispersion tensor in global (x, y) coordinates.

            D = R(θ) diag(D_L, D_T) R(θ)ᵀ
        """
        _, D_L, D_T, theta = self._params()
        c, s = torch.cos(theta), torch.sin(theta)
        D = torch.zeros(2, 2, dtype=D_L.dtype, device=D_L.device)
        D[0, 0] = D_L * c**2 + D_T * s**2
        D[1, 1] = D_L * s**2 + D_T * c**2
        D[0, 1] = D[1, 0] = (D_L - D_T) * s * c
        return D
