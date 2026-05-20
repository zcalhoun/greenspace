import unittest

# CLASSES
from greenspace.trainers import BayesianOptimization, Trial

# FUNCTIONS
from greenspace.trainers import _rand_sample_within_bounds

import numpy as np


class TestBayesianOptimization(unittest.TestCase):

    def test_initialize(self):
        bo = BayesianOptimization({"variable": (-5.0, 5.0)})
        self.assertIsInstance(bo, BayesianOptimization)
        self.assertIsInstance(bo.variables, dict)

    def test_get_next_trial(self):
        """
        Test that the trials are first random, then use GPs.
        """
        bo = BayesianOptimization({"parameter": (-5.0, 5.0)}, random_inits=1)
        next_trial = bo.get_next_trial()
        self.assertIsInstance(next_trial, Trial)
        self.assertIsInstance(next_trial.parameters, dict)
        self.assertIn("parameter", next_trial.parameters.keys())
        self.assertEqual(next_trial.type, "random")

        next_trial.update(-1.0)

        next_trial = bo.get_next_trial()
        next_trial.update(2.0)

        self.assertEqual(len(bo.trials), 2)
        self.assertEqual(next_trial.type, "ei")

    def test_rand_sample_bounds(self):
        """
        Confirm that the random sample is within the bounds expected.
        """
        bo = BayesianOptimization({"p1": (-5.0, 5.0)}, random_inits=1)

        self.assertTupleEqual(bo.variables["p1"], (-5.0, 5.0))

        for i in range(0, 100):
            val = _rand_sample_within_bounds((-5.0, 5.0))
            self.assertLess(val, 5.0)
            self.assertGreater(val, -5.0)

    def test_get_best_trial(self):
        """
        Test that the best trial is returned.
        """
        bo = BayesianOptimization({"parameter": [-5.0, 5.0]}, random_inits=1)
        trial1 = Trial({"parameter": 1})
        trial1.update(1)
        trial2 = Trial({"parameter": 2})
        trial2.update(0)
        bo.trials = [trial1, trial2]

        best_trial = bo.get_best_trial()
        self.assertDictEqual(best_trial.parameters, {"parameter": 2})
        self.assertEqual(best_trial.result, 0)
        self.assertIsInstance(best_trial, Trial)

    def test_collect_trials(self):
        """
        Test that collected trials yield two numpy arrays.

        """

        bo = BayesianOptimization({"p1": [-5.0, 5.0], "p2": [-1, 1]}, random_inits=1)
        trial1 = Trial({"p1": 1, "p2": 0.1})
        trial1.update(1)
        trial2 = Trial({"p1": 2, "p2": -0.1})
        trial2.update(0)
        bo.trials = [trial1, trial2]

        X, y = bo._collect_trials()

        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape, (2, 2))


class TestTrial(unittest.TestCase):

    def test_initialize(self):
        trial = Trial({"variable": 1.0})
        self.assertIsInstance(trial, Trial)

    def test_update(self):
        trial = Trial({"variable": 1.0})
        trial.update(1.0)
        self.assertEqual(trial.result, 1.0)


if __name__ == "__main__":
    unittest.main()
