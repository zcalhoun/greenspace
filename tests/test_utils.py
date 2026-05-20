import io
import contextlib
import unittest

import torch

from greenspace.utils import SimpleLogger, farthest_point_sample


class TestSimpleLogger(unittest.TestCase):

    def test_initialize(self):
        logger = SimpleLogger()
        self.assertEqual(logger.prefix, "LOG")

    def test_custom_prefix(self):
        logger = SimpleLogger(prefix="TEST")
        self.assertEqual(logger.prefix, "TEST")

    def _capture(self, fn):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn()
        return buf.getvalue()

    def test_log_contains_prefix(self):
        logger = SimpleLogger(prefix="MY")
        out = self._capture(lambda: logger.log("hello"))
        self.assertIn("MY", out)
        self.assertIn("hello", out)

    def test_warn_contains_warning(self):
        logger = SimpleLogger()
        out = self._capture(lambda: logger.warn("bad thing"))
        self.assertIn("WARNING", out)
        self.assertIn("bad thing", out)

    def test_error_contains_error(self):
        logger = SimpleLogger()
        out = self._capture(lambda: logger.error("boom"))
        self.assertIn("ERROR", out)
        self.assertIn("boom", out)

    def test_info_contains_info(self):
        logger = SimpleLogger()
        out = self._capture(lambda: logger.info("detail"))
        self.assertIn("INFO", out)
        self.assertIn("detail", out)

    def test_timer_logs_start_and_stop(self):
        logger = SimpleLogger()
        out = self._capture(lambda: (logger.start_timer("T"), logger.stop_timer("T")))
        self.assertIn("T", out)
        self.assertIn("Timer started", out)
        self.assertIn("Timer stopped", out)

    def test_timer_elapsed_is_non_negative(self):
        logger = SimpleLogger()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            logger.start_timer("X")
            logger.stop_timer("X")
        out = buf.getvalue()
        # Extract the elapsed-time value from the log line
        import re
        match = re.search(r"Elapsed time: ([\d.]+) minutes", out)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(float(match.group(1)), 0.0)


class TestFarthestPointSample(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.coords = torch.rand(50, 2)
        self.residuals = torch.rand(50)

    def test_output_shape(self):
        result = farthest_point_sample(self.coords, self.residuals, k=10)
        self.assertEqual(result.shape, (10, 2))

    def test_first_point_is_max_residual(self):
        result = farthest_point_sample(self.coords, self.residuals, k=5)
        expected_first = self.coords[self.residuals.argmax()]
        self.assertTrue(torch.allclose(result[0], expected_first))

    def test_k_equals_one(self):
        result = farthest_point_sample(self.coords, self.residuals, k=1)
        self.assertEqual(result.shape, (1, 2))

    def test_k_equals_n(self):
        result = farthest_point_sample(self.coords, self.residuals, k=50)
        self.assertEqual(result.shape, (50, 2))

    def test_points_are_from_input(self):
        result = farthest_point_sample(self.coords, self.residuals, k=10)
        for point in result:
            distances = ((self.coords - point) ** 2).sum(dim=1)
            self.assertTrue(distances.min().item() < 1e-6)


if __name__ == "__main__":
    unittest.main()
