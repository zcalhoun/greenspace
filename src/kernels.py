"""
Advection-Dispersion Equation (ADE) Kernel for GPyTorch
========================================================

Implements the steady-state ADE Green's function kernel in 1D and 2D,
derived via the SPSD inner product construction:

    k(x, x') = integral G(x; s) G(x'; s) ds

1D result:  k(x, x') = sigma^2 * exp(-|x - x'| / ell)
            where ell = 2D/v  (Matern-1/2 / OU kernel with physical parameterization)

2D result:  k(x, x') = sigma^2 * exp(v . (x - x') / 2D) * K0(kappa * r)
            where kappa = |v| / 2D,  r = |x - x'|
            (anisotropic Matern-0 kernel tilted along flow direction)
"""

import torch
import gpytorch
from gpytorch.kernels import Kernel
from torch import Tensor
import math


# ---------------------------------------------------------------------------
# 1D ADE Kernel
# ---------------------------------------------------------------------------


class ADEKernel1D(Kernel):
    """
    Steady-state 1D ADE Green's function kernel.

    Equivalent to the Matern-1/2 (OU) kernel with physically meaningful
    hyperparameters:

        k(x, x') = sigma^2 * exp(-v / (2D) * |x - x'|)

    Hyperparameters
    ---------------
    log_velocity : unconstrained, exponentiated to give v > 0
    log_diffusion : unconstrained, exponentiated to give D > 0
    log_outputscale : unconstrained, exponentiated to give sigma^2 > 0

    Physical interpretation
    -----------------------
    ell = 2D / v  is the dispersion length scale:
      - large D/v (diffusion-dominated) -> long-range spatial correlations
      - small D/v (advection-dominated) -> short memory, sharp gradients
    """

    has_lengthscale = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter("raw_velocity", torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter("raw_diffusion", torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter("raw_outputscale", torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_velocity", gpytorch.constraints.Positive())
        self.register_constraint("raw_diffusion", gpytorch.constraints.Positive())
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())

    @property
    def velocity(self) -> Tensor:
        return self.raw_velocity_constraint.transform(self.raw_velocity)

    @velocity.setter
    def velocity(self, value):
        self._set_transformed_param("raw_velocity", value)

    @property
    def diffusion(self) -> Tensor:
        return self.raw_diffusion_constraint.transform(self.raw_diffusion)

    @diffusion.setter
    def diffusion(self, value):
        self._set_transformed_param("raw_diffusion", value)

    @property
    def outputscale(self) -> Tensor:
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_transformed_param("raw_outputscale", value)

    @property
    def lengthscale(self) -> Tensor:
        """Physical dispersion length: ell = 2D / v"""
        return 2.0 * self.diffusion / self.velocity

    def forward(self, x1: Tensor, x2: Tensor, **params) -> Tensor:
        """
        Compute k(x1, x2) = sigma^2 * exp(-|x1 - x2| / ell)
        """
        # x1: (..., n, 1), x2: (..., m, 1)
        dist = self.covar_dist(x1, x2, square_dist=False)  # (..., n, m)
        ell = self.lengthscale  # (1,)
        return self.outputscale * torch.exp(-dist / ell)

    def _set_transformed_param(self, raw_name: str, value):
        constraint = getattr(self, f"{raw_name}_constraint")
        self.__getattr__(raw_name).data = constraint.inverse_transform(
            torch.as_tensor(value).to(self.__getattr__(raw_name))
        )


# ---------------------------------------------------------------------------
# 2D ADE Kernel
# ---------------------------------------------------------------------------


class ADEKernel2D(Kernel):
    """
    Steady-state 2D ADE Green's function kernel.

    Derived from the modified Helmholtz Green's function via the SPSD
    inner product construction:

        k(x, x') = sigma^2 * exp(v . (x - x') / 2D) * K0(kappa * r)

    where:
        kappa = |v| / 2D       (inverse dispersion length)
        r     = |x - x'|       (Euclidean distance)
        K0    = modified Bessel function of second kind, order 0

    The kernel is NOT stationary: it depends on the signed displacement
    (x - x'), not just |x - x'|. This encodes directional transport.

    Hyperparameters
    ---------------
    log_speed     : log of flow speed |v| > 0
    flow_angle    : flow direction theta in [0, 2*pi)
    log_diffusion : log of isotropic diffusion coefficient D > 0
    log_outputscale : log of amplitude sigma^2 > 0

    Input format
    ------------
    x1, x2 : Tensor of shape (..., n, 2) — (x, y) coordinates
    """

    has_lengthscale = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter("raw_speed", torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter("flow_angle", torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter("raw_diffusion", torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter("raw_outputscale", torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_speed", gpytorch.constraints.Positive())
        self.register_constraint("raw_diffusion", gpytorch.constraints.Positive())
        self.register_constraint("raw_outputscale", gpytorch.constraints.Positive())

    @property
    def speed(self) -> Tensor:
        return self.raw_speed_constraint.transform(self.raw_speed)

    @property
    def diffusion(self) -> Tensor:
        return self.raw_diffusion_constraint.transform(self.raw_diffusion)

    @property
    def outputscale(self) -> Tensor:
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @property
    def velocity_vector(self) -> Tensor:
        """v = speed * (cos theta, sin theta)"""
        theta = self.flow_angle
        return self.speed * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    @property
    def kappa(self) -> Tensor:
        """Inverse dispersion length kappa = |v| / 2D"""
        return self.speed / (2.0 * self.diffusion)

    def forward(self, x1: Tensor, x2: Tensor, **params) -> Tensor:
        """
        Compute k(x1, x2).

        k(x, x') = sigma^2 * exp(v . (x - x') / 2D) * K0(kappa * r)

        Numerically stabilized: K0(z) -> -log(z/2) as z -> 0, so we
        clamp r away from zero with a small epsilon.
        """
        v = self.velocity_vector  # (2,)
        D = self.diffusion  # scalar
        kappa = self.kappa  # scalar
        sigma2 = self.outputscale  # scalar

        # Signed displacement: (..., n, 1, 2) - (..., 1, m, 2)
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)  # (..., n, m, 2)

        # Advection factor: exp(v . delta_x / 2D)
        adv = torch.exp((diff * v).sum(dim=-1) / (2.0 * D))  # (..., n, m)

        # Euclidean distance with epsilon for numerical stability
        r = diff.norm(dim=-1).clamp(min=1e-6)  # (..., n, m)

        # Modified Bessel K0 via scipy-style approximation
        # Use torch's built-in special functions (torch >= 2.0)
        k0 = _bessel_k0(kappa * r)  # (..., n, m)

        return sigma2 * adv * k0

    def isotropy_ratio(self) -> float:
        """
        Ratio of along-flow to cross-flow correlation length.
        Values >> 1 indicate strong anisotropy (advection-dominated).
        """
        kappa_val = self.kappa.item()
        # Along-flow decay scale ~ 1/kappa, cross-flow ~ 1/kappa too for K0
        # but advection factor stretches correlation along v
        # Ratio approximated as exp(|v|^2 / (2D * kappa)) / 1
        v_mag = self.speed.item()
        D_val = self.diffusion.item()
        return math.exp(v_mag / (2.0 * D_val * kappa_val))


def _bessel_k0(z: Tensor) -> Tensor:
    """
    Modified Bessel function K0(z) for z > 0.

    Uses a rational polynomial approximation accurate to ~1e-7:
      - for z in (0, 2]: polynomial in (z/2)^2 - log(z/2) * I0(z)
      - for z > 2:       polynomial in (2/z) * exp(-z)

    Reference: Abramowitz & Stegun 9.8.5 / 9.8.6
    """
    z = z.clamp(min=1e-10)
    small = z <= 2.0

    # --- small z branch ---
    t_s = z[small] / 2.0
    t2 = t_s**2
    # I0 polynomial coefficients (A&S 9.8.1)
    i0_poly = 1.0 + t2 * (
        3.5156229
        + t2
        * (
            3.0899424
            + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))
        )
    )
    # K0 polynomial coefficients (A&S 9.8.5)
    k0_poly = -0.57721566 + t2 * (
        0.42278420
        + t2
        * (
            0.23069756
            + t2 * (0.03488590 + t2 * (0.00262698 + t2 * (0.00010750 + t2 * 0.0000074)))
        )
    )
    k0_small = -torch.log(t_s) * i0_poly + k0_poly

    # --- large z branch ---
    t_l = 2.0 / z[~small]
    k0_poly_l = 1.25331414 + t_l * (
        -0.07832358
        + t_l
        * (
            0.02189568
            + t_l
            * (
                -0.01062446
                + t_l * (0.00587872 + t_l * (-0.00251540 + t_l * 0.00053208))
            )
        )
    )
    k0_large = torch.exp(-z[~small]) / torch.sqrt(z[~small]) * k0_poly_l

    out = torch.empty_like(z)
    out[small] = k0_small
    out[~small] = k0_large
    return out


# ---------------------------------------------------------------------------
# Convenience: exact GP models using both kernels
# ---------------------------------------------------------------------------


class ExactGP_ADE1D(gpytorch.models.ExactGP):
    """
    Exact GP regression model with the 1D ADE kernel.

    Usage
    -----
    >>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = ExactGP_ADE1D(train_x, train_y, likelihood)
    >>> model.train(); likelihood.train()
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    >>> mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ADEKernel1D()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class ExactGP_ADE2D(gpytorch.models.ExactGP):
    """
    Exact GP regression model with the 2D ADE kernel.

    Input x should be shape (n, 2) with columns [x_coord, y_coord].
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ADEKernel2D()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


def train_gp(
    model,
    likelihood,
    train_x,
    train_y,
    n_iter: int = 200,
    lr: float = 0.05,
    verbose: bool = True,
):
    """
    Train a GPyTorch model via marginal log likelihood maximization.

    Returns the loss history.
    """
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and (i % 50 == 0 or i == n_iter - 1):
            print(f"  Iter {i+1:4d}/{n_iter} | Loss: {loss.item():.4f}")

    return losses


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("1D ADE kernel demo")
    print("=" * 60)

    # Synthetic data: decaying concentration with noise
    train_x = torch.linspace(0, 5, 20).unsqueeze(-1)
    true_v, true_D = 1.0, 0.5
    true_ell = 2 * true_D / true_v
    train_y = torch.exp(-train_x.squeeze() / true_ell) + 0.05 * torch.randn(20)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGP_ADE1D(train_x, train_y, likelihood)

    losses = train_gp(model, likelihood, train_x, train_y, n_iter=200, verbose=True)

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 6, 100).unsqueeze(-1)
        pred = likelihood(model(test_x))

    print(
        f"\nInferred velocity:  {model.covar_module.velocity.item():.3f}  (true: {true_v})"
    )
    print(
        f"Inferred diffusion: {model.covar_module.diffusion.item():.3f}  (true: {true_D})"
    )
    print(
        f"Inferred ell:       {model.covar_module.lengthscale.item():.3f}  (true: {true_ell})"
    )

    print("\n" + "=" * 60)
    print("2D ADE kernel demo")
    print("=" * 60)

    n_train = 30
    train_x2d = torch.rand(n_train, 2) * 4
    v_vec = torch.tensor([1.0, 0.5])
    D2d = 0.3
    kappa2d = v_vec.norm() / (2 * D2d)
    # Simple synthetic concentration field
    train_y2d = torch.exp(
        -(train_x2d * v_vec).sum(-1) / (2 * D2d)
    ) + 0.05 * torch.randn(n_train)

    likelihood2d = gpytorch.likelihoods.GaussianLikelihood()
    model2d = ExactGP_ADE2D(train_x2d, train_y2d, likelihood2d)

    losses2d = train_gp(
        model2d, likelihood2d, train_x2d, train_y2d, n_iter=200, verbose=True
    )

    model2d.eval()
    likelihood2d.eval()
    print(f"\nInferred speed:     {model2d.covar_module.speed.item():.3f}")
    print(f"Inferred angle:     {model2d.covar_module.flow_angle.item():.3f} rad")
    print(
        f"Inferred diffusion: {model2d.covar_module.diffusion.item():.3f}  (true: {D2d})"
    )
    print(f"Anisotropy ratio:   {model2d.covar_module.isotropy_ratio():.2f}")
