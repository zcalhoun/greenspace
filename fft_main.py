"""
This script implements Bayesian optimization to learn the parameters of the
AnisotropicADEGreen model (v, D_L, D_T ratio r, theta) that maximize R² when
predicting temperature from FFT-convolved NLCD land-cover features via Ridge
regression.

All that's needed is the landcover dataset and the traversal.
"""

import os
import sys
import argparse
import numpy as np
import math

import geopandas as gpd
import rasterio
from rasterio.transform import AffineTransformer
from pyproj import Transformer, Geod
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

import torch
import torch.nn.functional as F
from torch import nn


sys.path.append("./src")
from ade import AnisotropicADEGreen


def main(args):

    # LOAD THE DATA
    print("Loading data...")
    one_hot, coords, temp, nx, ny = load_data()
    print(f"  Grid size: {nx} x {ny}  |  Traversal points: {len(temp)}")

    # BAYESIAN OPTIMIZATION SETUP
    # We want to maximize R² of Ridge(NLCD features) → temperature.
    # Parameters being optimized:
    #   v     : advection speed (m/s equivalent)
    #   D_L   : longitudinal dispersion coefficient
    #   r     : D_T / D_L ratio (keeps D_T <= D_L physically)
    #   theta : flow direction (radians from +x axis)
    bounds = np.array([[0.01, 5.0], [1.0, 500.0], [0.1, 1.0], [0.0, 2 * np.pi]])

    # --- Initial random exploration ---
    print("\nEvaluating 10 random initial points...")
    X_obs = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(10, 4))
    y_obs = np.array([objective(x, one_hot, coords, temp, nx, ny) for x in X_obs])
    print(f"  Initial R² range: [{y_obs.min():.4f}, {y_obs.max():.4f}]")
    print(f"  Initial best R²:   {y_obs.max():.4f}")

    # Gaussian Process surrogate model over the parameter space
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale=np.ones(4), length_scale_bounds=(1e-3, 1e4)),
        normalize_y=True,
        n_restarts_optimizer=5,
    )

    # --- Bayesian optimization loop ---
    print("\nStarting Bayesian optimization (50 iterations)...")
    for i in range(50):
        gp.fit(X_obs, y_obs)

        # Sample 1000 random candidates and score them with the EI acquisition.
        # We use EI for MAXIMIZATION: prefer regions predicted to exceed the
        # current best R² by at least xi.
        X_cand = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1000, 4))
        ei = expected_improvement(X_cand, gp, f_best=y_obs.max())
        x_next = X_cand[ei.argmax()]

        y_next = objective(x_next, one_hot, coords, temp, nx, ny)
        X_obs = np.append(X_obs, [x_next], axis=0)
        y_obs = np.append(y_obs, y_next)

        print(
            f"  Iter {i+1:3d} | R²={y_next:.4f} | best so far={y_obs.max():.4f} | "
            f"v={x_next[0]:.3f} D_L={x_next[1]:.1f} r={x_next[2]:.3f} θ={x_next[3]:.3f}"
        )

    # Pick the parameter set that achieved the highest R²
    best_idx = y_obs.argmax()
    best = X_obs[best_idx]
    v, D_L, r, theta = best
    D_T = D_L * r

    print("\n=== Optimization complete ===")
    print(f"  Best R²:  {y_obs[best_idx]:.4f}")
    print(f"  v={v:.4f}  D_L={D_L:.4f}  D_T={D_T:.4f}  theta={theta:.4f} rad")


def expected_improvement(X, gp, f_best, xi=0.01):
    """
    Expected Improvement acquisition function for MAXIMIZATION.

    Scores each candidate by the expected amount it will exceed f_best.
    Higher EI → more promising candidate.

    Parameters
    ----------
    X      : (N, d) array of candidate points
    gp     : fitted GaussianProcessRegressor
    f_best : current best observed objective value (maximum R² so far)
    xi     : exploration bonus — larger values encourage more exploration
    """
    mu, sigma = gp.predict(X, return_std=True)
    # improvement > 0 when the predicted mean exceeds the current best
    improvement = mu - f_best - xi
    z = improvement / (sigma + 1e-9)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    # EI is 0 wherever the surrogate has no uncertainty (avoid numeric noise)
    ei[sigma < 1e-10] = 0.0
    return ei


def objective(params, S, coords, obs, nx, ny):
    """
    Evaluate R² for a given set of ADE parameters.

    Builds the Green's function kernel, convolves it with the one-hot NLCD
    source field S, then fits a Ridge regression at the traversal coordinates
    and returns R² against the observed temperatures.

    Parameters
    ----------
    params : (v, D_L, r, theta)
    S      : (C, nx, ny) one-hot NLCD tensor
    coords : (N, 2) integer pixel coordinates of traversal points
    obs    : (N,) observed temperatures
    nx, ny : spatial grid dimensions

    Returns
    -------
    float : R² (higher is better; Bayesian optimization MAXIMIZES this)
    """
    v, D_L, r, theta = params
    D_T = D_L * r
    model = AnisotropicADEGreen(
        nx,
        ny,
        dx=30.0,
        dy=30.0,
        v=v,
        D_L=D_L,
        D_T=D_T,
        theta=theta,
        # padding ≈ 3 longitudinal decay lengths to avoid wrap-around artifacts
        padding=10,  # math.ceil(3 * D_L / (v * 30.0)),
    )
    with torch.no_grad():
        res = model(S)  # (C, nx, ny)

    # Extract the convolved feature vector at each traversal point: shape (N, C)
    X = res[:, coords[:, 0], coords[:, 1]].T

    X = np.column_stack([X, S[:, coords[:, 0], coords[:, 1]].T])
    lm = Ridge(0.01, positive=False, fit_intercept=False)
    lm.fit(X, obs - obs.mean())
    r2 = lm.score(X, obs - obs.mean())
    return r2


def load_data():

    with rasterio.open("./data/nlcd/durham.tif") as src:
        data = src.read(1)
        nlcd_crs = src.crs
        nlcd_transform = src.transform

    _NLCD_CLASSES = [
        11,
        12,
        21,
        22,
        23,
        24,
        31,
        41,
        42,
        43,
        51,
        52,
        71,
        72,
        73,
        74,
        81,
        82,
        90,
        95,
    ]

    _NLCD_LUT = torch.full(
        (max(_NLCD_CLASSES) + 2,), len(_NLCD_CLASSES), dtype=torch.long
    )

    for _idx, _val in enumerate(_NLCD_CLASSES):
        _NLCD_LUT[_val] = _idx

    d = torch.from_numpy(data).long()

    nlcd_oh = _NLCD_LUT[d.clamp(0, _NLCD_LUT.size(0) - 1)]

    # One-hot encode: (num_classes+1, nx, ny)
    oh = F.one_hot(nlcd_oh, num_classes=len(_NLCD_CLASSES) + 1).float().permute(2, 0, 1)

    # Raster grid dimensions needed by AnisotropicADEGreen
    _, nx, ny = oh.shape

    gdf = gpd.read_file(os.path.join("./data/traversals/durham/pm/trav.shp"))
    gdf_reproj = gdf.to_crs(nlcd_crs)
    transformer = AffineTransformer(nlcd_transform)
    nlcd_coords = np.column_stack(
        transformer.rowcol(gdf_reproj.geometry.x, gdf_reproj.geometry.y)
    )

    y = gdf["temp_f"].values

    return oh, nlcd_coords, y, nx, ny


if __name__ == "__main__":
    main(None)
