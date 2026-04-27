"""
Isotropic exponential convolution kernel.

K(r) = exp(-r / L)

where r is the Euclidean distance from each source pixel and L is the
length scale in the same units as dx/dy.  The kernel is built in spatial
space, FFT'd, and used identically to AnisotropicADEGreen so the two can
be swapped in the objective function without any other changes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class IsotropicExponential(nn.Module):
    """
    FFT convolution with an isotropic exponential decay kernel.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    dx, dy : float
        Physical grid spacing (same units as L).
    L : float
        Exponential length scale.
    padding : int
        Zero-padding cells added to each side before FFT.
    normalize : bool
        If True the kernel weights sum to 1 (weighted average).
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float = 1.0,
        dy: float = 1.0,
        L: float = 1.0,
        padding: int = 0,
        normalize: bool = True,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.pad = padding

        nx_pad = nx + 2 * padding
        ny_pad = ny + 2 * padding

        # Build distance grid with FFT-convention origin at (0, 0).
        ix = torch.arange(nx_pad, dtype=torch.float64)
        iy = torch.arange(ny_pad, dtype=torch.float64)
        ix = torch.where(ix < nx_pad // 2, ix, ix - nx_pad) * dx
        iy = torch.where(iy < ny_pad // 2, iy, iy - ny_pad) * dy
        IX, IY = torch.meshgrid(ix, iy, indexing="ij")
        r = torch.sqrt(IX ** 2 + IY ** 2)

        kernel = torch.exp(-r / L)
        G_hat = torch.fft.fft2(kernel)

        if normalize:
            G_hat = G_hat / G_hat[0, 0].real

        self.register_buffer("G_hat", G_hat)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        S : Tensor, shape (..., nx, ny)

        Returns
        -------
        Tensor, shape (..., nx, ny)
        """
        if self.pad > 0:
            S = F.pad(S, (self.pad, self.pad, self.pad, self.pad))

        C_pad = torch.fft.ifft2(torch.fft.fft2(S) * self.G_hat).real

        if self.pad > 0:
            C_pad = C_pad[
                ...,
                self.pad : self.pad + self.nx,
                self.pad : self.pad + self.ny,
            ]
        return C_pad
