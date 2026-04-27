"""
Baseline experiment: isotropic exponential convolution kernel.

Optimises a single length scale L via Bayesian optimisation, then fits
the same Ridge regression as fft_main.py.  Results saved alongside the
ADE results for direct comparison.
"""

import os
import sys
import json
import warnings
import numpy as np

import geopandas as gpd
import rasterio
from rasterio.transform import AffineTransformer
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

import torch
import torch.nn.functional as F


sys.path.append("./src")
from iso_exp import IsotropicExponential


PERIODS = ["am", "af", "pm"]
TRAVERSAL_ROOT = "./data/traversals"
LCZ_ROOT = "./data/lcz"
OUT_DIR = "./results/lcz_iso"

# BO searches log(L) so the surrogate sees a well-scaled 1-D space.
# L range: 100 m – 50 km  →  log bounds below.
LOG_L_BOUNDS = np.array([[np.log(100.0), np.log(50_000.0)]])
OFFSET = 0.0

N_INIT = 10
N_ITER = 100


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cities = sorted(os.listdir(TRAVERSAL_ROOT))

    for city in cities:
        for period in PERIODS:
            trav_shp = os.path.join(TRAVERSAL_ROOT, city, period, "trav.shp")
            lcz_tif  = os.path.join(LCZ_ROOT, f"{city}_lcz.tif")
            out_path = os.path.join(OUT_DIR, f"{city}_{period}.json")

            if not os.path.exists(trav_shp):
                print(f"[skip] {city}/{period} — no traversal file")
                continue
            if not os.path.exists(lcz_tif):
                print(f"[skip] {city}/{period} — no LCZ raster")
                continue
            if os.path.exists(out_path):
                print(f"[skip] {city}/{period} — already done")
                continue

            print(f"\n{'='*60}")
            print(f"  {city.upper()}  |  {period.upper()}")
            print(f"{'='*60}")

            try:
                run_city_period(city, period, trav_shp, lcz_tif, out_path)
            except Exception as e:
                print(f"  ERROR: {e}")


def run_city_period(city, period, trav_shp, lcz_tif, out_path):
    one_hot, coords, temp, nx, ny = load_data(trav_shp, lcz_tif)
    print(f"  Grid: {nx}x{ny}  |  Points: {len(temp)}")

    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-3, 1e4)),
        normalize_y=True,
        n_restarts_optimizer=5,
        alpha=0.001,
    )

    print(f"  Evaluating {N_INIT} random initial points...")
    log_L_obs = np.random.uniform(
        LOG_L_BOUNDS[0, 0], LOG_L_BOUNDS[0, 1], size=(N_INIT, 1)
    )
    y_obs = np.array([objective(np.exp(x[0]), one_hot, coords, temp, nx, ny) for x in log_L_obs])
    print(f"  Initial R² range: [{y_obs.min():.4f}, {y_obs.max():.4f}]")

    print(f"  Running {N_ITER} BO iterations...")
    for i in range(N_ITER):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*length_scale.*", category=UserWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            gp.fit(log_L_obs, y_obs)

        cands = np.random.uniform(
            LOG_L_BOUNDS[0, 0], LOG_L_BOUNDS[0, 1], size=(1000, 1)
        )
        ei = expected_improvement(cands, gp, f_best=y_obs.max())
        log_L_next = cands[ei.argmax()]

        y_next = objective(np.exp(log_L_next[0]), one_hot, coords, temp, nx, ny)
        log_L_obs = np.append(log_L_obs, [log_L_next], axis=0)
        y_obs = np.append(y_obs, y_next)

        if (i + 1) % 10 == 0:
            print(f"  Iter {i+1:3d} | best R²={y_obs.max():.4f}")

    best_idx = y_obs.argmax()
    best_r2 = float(y_obs[best_idx])
    L = float(np.exp(log_L_obs[best_idx, 0]))
    coefs = fit_final_model(L, one_hot, coords, temp, nx, ny)

    result = {
        "city": city,
        "period": period,
        "r2": best_r2,
        "params": {"L": L, "offset": OFFSET},
        "coefficients": coefs.tolist(),
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Done. R²={best_r2:.4f}  →  {out_path}")


def fit_final_model(L, S, coords, obs, nx, ny):
    model = IsotropicExponential(nx, ny, dx=100.0, dy=100.0, L=L, padding=10)
    with torch.no_grad():
        res = model(S)
    X = res[:, coords[:, 0], coords[:, 1]].T
    lm = Ridge(0.01, positive=True, fit_intercept=False)
    lm.fit(X, obs - obs.min() - OFFSET)
    return lm.coef_


def objective(L, S, coords, obs, nx, ny):
    model = IsotropicExponential(nx, ny, dx=100.0, dy=100.0, L=L, padding=10)
    with torch.no_grad():
        res = model(S)
    X = res[:, coords[:, 0], coords[:, 1]].T
    lm = Ridge(0.01, positive=True, fit_intercept=False)
    lm.fit(X, obs - obs.min() - OFFSET)
    return lm.score(X, obs - obs.min() - OFFSET)


def expected_improvement(X, gp, f_best, xi=0.01):
    mu, sigma = gp.predict(X, return_std=True)
    improvement = mu - f_best - xi
    z = improvement / (sigma + 1e-9)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-10] = 0.0
    return ei


def load_data(trav_shp, lcz_tif):
    with rasterio.open(lcz_tif) as src:
        data = src.read(1)
        lcz_crs = src.crs
        lcz_transform = src.transform

    oh = (
        F.one_hot(torch.from_numpy(data).long(), num_classes=18)
        .float()
        .permute(2, 0, 1)
    )
    _, nx, ny = oh.shape

    gdf = gpd.read_file(trav_shp).to_crs(lcz_crs)
    transformer = AffineTransformer(lcz_transform)
    coords = np.column_stack(
        transformer.rowcol(gdf.geometry.x, gdf.geometry.y)
    )

    temp_col = next(c for c in ("temp_f", "t_f", "T") if c in gdf.columns)
    temp = gdf[temp_col].values

    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < nx) &
        (coords[:, 1] >= 0) & (coords[:, 1] < ny)
    )
    coords, temp = coords[valid], temp[valid]

    return oh, coords, temp, nx, ny


if __name__ == "__main__":
    main()
