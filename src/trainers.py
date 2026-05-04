"""
This is my lightweight implementation of Bayesian optimization.

"""

from __future__ import annotations
from itertools import product

import numpy as np
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor, kernels


class BayesianOptimization:
    """
    A basic implementation of Bayesian Optimization that keeps track of performance,
    and bounds to find the next value.
    """

    def __init__(
        self,
        variables: dict[str, tuple],
        minimize: bool = True,
        random_inits: int = 5,
        dim_size=100,
    ):
        self.variables = variables
        self.trials = []
        self.random_inits = random_inits
        self.minimize = minimize
        self.X_candidates = self._create_grid(dim_size)

    def _create_grid(self, dim_size) -> np.ndarray:

        grids = [
            np.linspace(start, stop, dim_size)
            for start, stop in self.variables.values()
        ]
        cp = list(product(*grids))  # Obtain the cross product

        return cp

    def get_next_trial(self):
        if len(self.trials) < self.random_inits:
            # Randomly sample from the bounds
            parameters = {}
            for v in self.variables.keys():
                parameters[v] = _rand_sample_within_bounds(self.variables[v])

            next_trial = Trial(parameters, trial_type="random")
        else:
            X, y = self._collect_trials()

            f_best = self.get_best_trial().result

            gp = _fit_gp(X, y)

            ei = expected_improvement(self.X_candidates, gp, f_best)

            X_next = self.X_candidates[np.argmax(ei)]
            next_trial = Trial(
                {v: X_next[i] for i, v in enumerate(self.variables.keys())},
                trial_type="ei",
            )

        self.trials.append(next_trial)
        return next_trial

    def _collect_trials(self):
        X = [list(t.parameters.values()) for t in self.trials]
        y = [t.result for t in self.trials]

        return np.array(X), np.array(y)

    def get_best_trial(self) -> Trial:
        """
        Return the trial instance that has the best performance.
        """
        results = [t.result for t in self.trials]

        if self.minimize:
            return self.trials[np.argmin(results)]
        else:
            return self.trials[np.argmax(results)]


def expected_improvement(X, gp, f_best):
    mu, sigma = gp.predict(X, return_std=True)
    Z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0  # no improvement possible where variance is ~0
    return ei


def _fit_gp(X, y):
    k = 0.5**2 * kernels.RBF(length_scale=np.ones(X.shape[1]))
    gp = GaussianProcessRegressor(k, normalize_y=True)
    gp.fit(X, y)
    return gp


def _rand_sample_within_bounds(bounds: tuple) -> float:
    return np.random.uniform(bounds[0], bounds[1])


class Trial:
    """
    A simple class to hold a set of parameters and results.
    """

    def __init__(self, parameters, trial_type="random"):
        self.parameters = parameters
        self.type = trial_type

    def update(self, result):
        self.result = result
