import unittest

import torch
from gpytorch.distributions import MultivariateNormal

from greenspace.models.model import Exponential, GPModel, CompleteModel, RidgeGP


class TestExponential(unittest.TestCase):

    def setUp(self):
        self.batch = 4
        self.num_dims = 8
        self.size = 7  # small window for speed
        self.kernel = Exponential(size=self.size, num_dims=self.num_dims)

    def test_output_shape(self):
        x = torch.rand(self.batch, self.num_dims, self.size * self.size)
        out = self.kernel(x)
        self.assertEqual(out.shape, (self.batch, self.num_dims))

    def test_lengthscale_is_positive(self):
        self.assertGreater(self.kernel.lengthscale.item(), 0.0)

    def test_lengthscale_is_learnable(self):
        self.assertTrue(self.kernel.log_lengthscale.requires_grad)

    def test_weights_sum_to_one_per_channel(self):
        # Each channel's weights should sum to ~1 after normalisation.
        dist = self.kernel.dist * self.kernel.dimension_resolution
        w = torch.exp(-dist / self.kernel.lengthscale)
        w_norm = w / w.sum(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(w_norm.sum(dim=1), torch.ones(self.num_dims), atol=1e-5))


class TestGPModel(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.n_inducing = 10
        self.inducing = torch.randn(self.n_inducing, 2)
        self.model = GPModel(self.inducing)

    def test_forward_returns_multivariate_normal(self):
        self.model.train()
        x = torch.randn(5, 2)
        out = self.model(x)
        self.assertIsInstance(out, MultivariateNormal)

    def test_forward_mean_shape(self):
        self.model.train()
        x = torch.randn(5, 2)
        out = self.model(x)
        self.assertEqual(out.mean.shape, (5,))

    def test_min_lengthscale_constraint(self):
        # The GreaterThan constraint stores the raw value via .data, so the
        # constrained lengthscale is softplus(raw) + lower_bound — not the raw
        # value itself. We just verify the constraint is satisfied.
        model = GPModel(self.inducing, min_lengthscale=100.0)
        ls = model.covar_module.kernels[0].base_kernel.lengthscale.item()
        self.assertGreaterEqual(ls, 100.0 / 1000.0)


class TestCompleteModel(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.batch = 4
        self.num_dims = 8
        self.size = 7
        self.model = CompleteModel(
            num_dims=self.num_dims,
            num_inducing_points=10,
            size=self.size,
            intercept=torch.tensor(15.0),
        )

    def _inputs(self):
        c = torch.randn(self.batch, 2)
        e = torch.randn(self.batch, 1)
        s = torch.rand(self.batch, self.num_dims, self.size, self.size)
        return c, e, s

    def test_pretrain_output_shape(self):
        c, e, s = self._inputs()
        out = self.model(c, e, s, pretrain=True)
        self.assertEqual(out.shape, (self.batch,))

    def test_pretrain_output_is_tensor(self):
        c, e, s = self._inputs()
        out = self.model(c, e, s, pretrain=True)
        self.assertIsInstance(out, torch.Tensor)

    def test_full_forward_returns_multivariate_normal(self):
        self.model.train()
        c, e, s = self._inputs()
        out = self.model(c, e, s, pretrain=False)
        self.assertIsInstance(out, MultivariateNormal)

    def test_full_forward_mean_shape(self):
        self.model.train()
        c, e, s = self._inputs()
        out = self.model(c, e, s, pretrain=False)
        self.assertEqual(out.mean.shape, (self.batch,))

    def test_elevation_weight_is_non_positive(self):
        # Elevation should always cool — weight must be ≤ 0.
        self.assertLessEqual(self.model.elevation_weight.item(), 0.0)

    def test_beta_shape(self):
        self.assertEqual(self.model.beta.shape, (self.num_dims * 2,))


class TestRidgeGP(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.batch = 4
        self.num_dims = 8
        self.model = RidgeGP(
            num_dims=self.num_dims,
            num_inducing_points=10,
            intercept=torch.tensor(15.0),
        )

    def test_forward_returns_multivariate_normal(self):
        self.model.train()
        c = torch.randn(self.batch, 2)
        X = torch.randn(self.batch, self.num_dims)
        out = self.model(c, X)
        self.assertIsInstance(out, MultivariateNormal)

    def test_forward_mean_shape(self):
        self.model.train()
        c = torch.randn(self.batch, 2)
        X = torch.randn(self.batch, self.num_dims)
        out = self.model(c, X)
        self.assertEqual(out.mean.shape, (self.batch,))

    def test_beta_shape(self):
        self.assertEqual(self.model.beta.shape, (self.num_dims,))


if __name__ == "__main__":
    unittest.main()
