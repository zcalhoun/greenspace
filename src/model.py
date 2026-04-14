import torch
from torch import nn

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.distributions import MultivariateNormal


class Exponential(nn.Module):
    def __init__(
        self, size=51, lengthscale=500.0, num_dims=31, dimension_resolution=None
    ):
        """
        Args:
            size:                 window side length in pixels
            lengthscale:          initial lengthscale in metres (single learned scalar,
                                  stored in log-space to keep it strictly positive)
            num_dims:             number of feature channels
            dimension_resolution: (num_dims,) tensor of metres-per-pixel for each
                                  channel; defaults to 1.0 (pixel units) if None
        """
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.tensor(float(lengthscale)).log())

        idxs = torch.arange(0 - size // 2, size // 2 + 1, dtype=torch.float)
        points = torch.meshgrid(idxs, idxs, indexing="ij")
        dist = torch.sqrt(points[0] ** 2 + points[1] ** 2)
        dist[size // 2, size // 2] = 1e6
        dist = dist.ravel()
        self.register_buffer("dist", dist)

        if dimension_resolution is not None:
            res = dimension_resolution.float().view(num_dims, 1)
        else:
            res = torch.ones(num_dims, 1)
        self.register_buffer("dimension_resolution", res)

    @property
    def lengthscale(self):
        return self.log_lengthscale.exp()

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_features, size * size]
        """
        # Scale pixel distances to metres per channel → (num_dims, size*size)
        dist_metres = self.dist * self.dimension_resolution
        w = torch.exp(-dist_metres / self.lengthscale)
        w = w / w.sum(dim=1, keepdim=True)
        return (x * w).sum(dim=2)


class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5)
        ) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class CompleteModel(nn.Module):
    """
    This model is going to take care of creating the linear terms and adding
    to the GP Layer to fit the complete model.
    """

    def __init__(
        self,
        num_dims=31,
        num_inducing_points=100,
        size=51,
        intercept=torch.tensor(15.0),
        dimension_resolution=None,
    ):
        super().__init__()

        self.elevation_weight = nn.Parameter(torch.tensor([-0.01]))
        self.beta = nn.Parameter(torch.randn(num_dims * 2))
        self.intercept = nn.Parameter(intercept)
        self.exp_weight = Exponential(
            size=size, num_dims=num_dims, dimension_resolution=dimension_resolution
        )
        self.gp_layer = GPModel(inducing_points=torch.randn(num_inducing_points, 2))

        self.size = size

    def forward(self, c, e, s, pretrain=False):
        # c is [batch_size, 2]
        # e is [batch_size, 1]
        # s is [batch_size, num_features, size, size]

        point = s[:, :, self.size // 2, self.size // 2]
        window_flat = s.flatten(start_dim=2)

        # Get the linear terms
        linear_terms = self.exp_weight(window_flat)

        # Concatenate elevation, points, linear terms
        x = torch.cat([point, linear_terms], dim=1)

        mu = e @ self.elevation_weight + x @ self.beta + self.intercept

        if pretrain:
            return mu

        # Get the GP output
        gp_output = self.gp_layer(c)

        new_mu = mu + gp_output.mean

        return MultivariateNormal(new_mu, gp_output.lazy_covariance_matrix)
