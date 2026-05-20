import unittest

import numpy as np
import torch

from greenspace.dataloaders.dataset import (
    _next_fast_fft_size,
    _fft_filter_exp,
    _one_hot_pool_raster,
)


class TestNextFastFftSize(unittest.TestCase):

    def test_exact_power_of_two_unchanged(self):
        for p in [1, 2, 4, 8, 16, 64, 128, 256]:
            self.assertEqual(_next_fast_fft_size(p), p)

    def test_rounds_up_to_next_power_of_two(self):
        self.assertEqual(_next_fast_fft_size(3), 4)
        self.assertEqual(_next_fast_fft_size(5), 8)
        self.assertEqual(_next_fast_fft_size(100), 128)
        self.assertEqual(_next_fast_fft_size(129), 256)

    def test_result_is_always_gte_input(self):
        for n in range(1, 200):
            self.assertGreaterEqual(_next_fast_fft_size(n), n)

    def test_result_is_always_power_of_two(self):
        for n in range(1, 200):
            p = _next_fast_fft_size(n)
            self.assertEqual(p & (p - 1), 0)


class TestFftFilterExp(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.C, self.H, self.W = 3, 32, 32
        self.pixel_size_m = 10.0
        self.lengthscale = 50.0

    def test_output_shape_preserved(self):
        x = torch.rand(self.C, self.H, self.W)
        out = _fft_filter_exp(x, self.pixel_size_m, self.lengthscale)
        self.assertEqual(out.shape, (self.C, self.H, self.W))

    def test_output_is_contiguous(self):
        x = torch.rand(self.C, self.H, self.W)
        out = _fft_filter_exp(x, self.pixel_size_m, self.lengthscale)
        self.assertTrue(out.is_contiguous())

    def test_non_negative_input_gives_non_negative_output(self):
        # The exponential kernel is strictly positive, so a non-negative raster
        # should produce a non-negative output.
        x = torch.rand(1, 16, 16)
        out = _fft_filter_exp(x, self.pixel_size_m, self.lengthscale)
        self.assertTrue((out >= -1e-5).all())

    def test_smoothing_reduces_variance(self):
        # Filtering a noisy raster with a long lengthscale should reduce variance.
        torch.manual_seed(42)
        x = torch.rand(1, 64, 64)
        out = _fft_filter_exp(x, pixel_size_m=1.0, lengthscale=20.0)
        self.assertLess(out.var().item(), x.var().item())

    def test_single_channel(self):
        x = torch.rand(1, self.H, self.W)
        out = _fft_filter_exp(x, self.pixel_size_m, self.lengthscale)
        self.assertEqual(out.shape, (1, self.H, self.W))


class TestOneHotPoolRaster(unittest.TestCase):

    def _make_lut(self, classes):
        lut = torch.full((max(classes) + 2,), len(classes), dtype=torch.long)
        for idx, val in enumerate(classes):
            lut[val] = idx
        return lut

    def test_output_shape_no_pooling(self):
        classes = [1, 2, 3]
        lut = self._make_lut(classes)
        data = np.random.choice(classes, size=(16, 16)).astype(np.int32)
        out = _one_hot_pool_raster(data, lut, num_classes=3, pool_factor=1)
        self.assertEqual(out.shape, (3, 16, 16))

    def test_output_shape_with_pooling(self):
        classes = [1, 2, 3]
        lut = self._make_lut(classes)
        data = np.random.choice(classes, size=(16, 16)).astype(np.int32)
        out = _one_hot_pool_raster(data, lut, num_classes=3, pool_factor=4)
        self.assertEqual(out.shape, (3, 4, 4))

    def test_values_in_zero_one_range(self):
        classes = [1, 2, 3]
        lut = self._make_lut(classes)
        data = np.random.choice(classes, size=(16, 16)).astype(np.int32)
        out = _one_hot_pool_raster(data, lut, num_classes=3, pool_factor=1)
        self.assertTrue((out >= 0).all())
        self.assertTrue((out <= 1).all())

    def test_class_fractions_sum_to_one_without_pooling(self):
        # With no pooling, each pixel is one-hot so classes sum to 1 per pixel.
        classes = [1, 2, 3]
        lut = self._make_lut(classes)
        data = np.random.choice(classes, size=(16, 16)).astype(np.int32)
        out = _one_hot_pool_raster(data, lut, num_classes=3, pool_factor=1)
        pixel_sums = out.sum(dim=0)
        self.assertTrue(torch.allclose(pixel_sums, torch.ones(16, 16), atol=1e-5))

    def test_uniform_raster_gives_single_hot_channel(self):
        # A raster containing only class 2 → channel 1 is all-ones, rest zero.
        classes = [1, 2, 3]
        lut = self._make_lut(classes)
        data = np.full((8, 8), 2, dtype=np.int32)
        out = _one_hot_pool_raster(data, lut, num_classes=3, pool_factor=1)
        self.assertTrue(torch.allclose(out[1], torch.ones(8, 8)))
        self.assertTrue(torch.allclose(out[0], torch.zeros(8, 8)))
        self.assertTrue(torch.allclose(out[2], torch.zeros(8, 8)))

    def test_unknown_class_maps_to_zeros(self):
        # Pixels with values not in the class list should produce all-zero encoding.
        classes = [1, 2, 3]
        lut = self._make_lut(classes)
        data = np.full((4, 4), 99, dtype=np.int32)  # 99 is not in classes
        out = _one_hot_pool_raster(data, lut, num_classes=3, pool_factor=1)
        self.assertTrue(torch.allclose(out, torch.zeros(3, 4, 4)))


if __name__ == "__main__":
    unittest.main()
