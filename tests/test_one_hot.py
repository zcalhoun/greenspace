import sys
import pytest
import torch

sys.path.insert(0, "./src")
from dataloader_v2 import GreenspaceDataset, _NLCD_CLASSES, _GREEN_CLASSES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def one_hot(window: torch.Tensor, lut: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Thin wrapper around the method under test."""
    # Instantiate a throwaway dataset object to access _create_one_hot without
    # hitting __init__ (which requires real files on disk).
    obj = object.__new__(GreenspaceDataset)
    return obj._create_one_hot(window, lut, num_classes)


def make_window(*values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long).unsqueeze(0)  # (1, N)


# ---------------------------------------------------------------------------
# NLCD tests
# ---------------------------------------------------------------------------

class TestNLCDOneHot:
    lut = GreenspaceDataset._NLCD_LUT
    n = len(_NLCD_CLASSES)

    def _oh(self, *values):
        return one_hot(make_window(*values), self.lut, self.n)

    def test_output_shape(self):
        out = self._oh(11, 21)
        assert out.shape == (1, 2, self.n)

    def test_valid_first_class(self):
        # NLCD 11 is the first class → channel 0 should be hot
        out = self._oh(11)
        assert out[0, 0, 0] == 1.0
        assert out[0, 0, 1:].sum() == 0.0

    def test_valid_last_class(self):
        # NLCD 95 is the last class → last channel should be hot
        out = self._oh(95)
        assert out[0, 0, self.n - 1] == 1.0
        assert out[0, 0, : self.n - 1].sum() == 0.0

    def test_valid_mid_class(self):
        # NLCD 41 is index 7 in _NLCD_CLASSES
        expected_idx = _NLCD_CLASSES.index(41)
        out = self._oh(41)
        assert out[0, 0, expected_idx] == 1.0
        assert out[0, 0].sum() == 1.0

    def test_invalid_value_within_lut_bounds(self):
        # 25 is within [0, 95] but not a valid NLCD class → all zeros
        out = self._oh(25)
        assert out[0, 0].sum() == 0.0

    def test_invalid_value_above_lut_max(self):
        # 200 is above max(NLCD_CLASSES)=95 → clamped → all zeros
        out = self._oh(200)
        assert out[0, 0].sum() == 0.0

    def test_invalid_value_zero(self):
        # 0 is not a valid NLCD class → all zeros
        out = self._oh(0)
        assert out[0, 0].sum() == 0.0

    def test_mixed_valid_and_invalid(self):
        # Window with one valid and one invalid pixel
        out = self._oh(11, 999)
        assert out[0, 0, 0] == 1.0   # pixel 0: valid, class 11 → channel 0
        assert out[0, 1].sum() == 0.0  # pixel 1: invalid → all zeros

    def test_all_valid_classes(self):
        # Every known class should produce exactly one hot channel
        for val in _NLCD_CLASSES:
            out = self._oh(val)
            assert out[0, 0].sum() == 1.0, f"NLCD class {val} did not produce a one-hot vector"

    def test_output_is_float(self):
        out = self._oh(11)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Greenspace tests
# ---------------------------------------------------------------------------

class TestGreenspaceOneHot:
    lut = GreenspaceDataset._GS_LUT
    n = len(_GREEN_CLASSES)

    def _oh(self, *values):
        return one_hot(make_window(*values), self.lut, self.n)

    def test_output_shape(self):
        out = self._oh(0, 2)
        assert out.shape == (1, 2, self.n)

    def test_valid_first_class(self):
        # GS class 0 → channel 0
        out = self._oh(0)
        assert out[0, 0, 0] == 1.0
        assert out[0, 0, 1:].sum() == 0.0

    def test_valid_last_class(self):
        # GS class 14 is the last → last channel hot
        out = self._oh(14)
        assert out[0, 0, self.n - 1] == 1.0
        assert out[0, 0, : self.n - 1].sum() == 0.0

    def test_valid_mid_class(self):
        expected_idx = _GREEN_CLASSES.index(8)
        out = self._oh(8)
        assert out[0, 0, expected_idx] == 1.0
        assert out[0, 0].sum() == 1.0

    def test_invalid_value_within_lut_bounds(self):
        # 1 is within [0, 14] but not in _GREEN_CLASSES → all zeros
        out = self._oh(1)
        assert out[0, 0].sum() == 0.0

    def test_invalid_value_above_lut_max(self):
        out = self._oh(999)
        assert out[0, 0].sum() == 0.0

    def test_mixed_valid_and_invalid(self):
        out = self._oh(2, 3)
        assert out[0, 0].sum() == 1.0   # 2 is valid
        assert out[0, 1].sum() == 0.0   # 3 is not a valid GS class

    def test_all_valid_classes(self):
        for val in _GREEN_CLASSES:
            out = self._oh(val)
            assert out[0, 0].sum() == 1.0, f"GS class {val} did not produce a one-hot vector"

    def test_output_is_float(self):
        out = self._oh(4)
        assert out.dtype == torch.float32
