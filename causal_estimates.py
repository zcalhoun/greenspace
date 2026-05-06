import os
import argparse
import sys

import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import transform_bounds
from rasterio.transform import AffineTransformer
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from linear_operator.utils.errors import NotPSDError

sys.path.append("./src")
from dataloader_v2 import CausalGreenspaceDataset
from model import RidgeGP
from utils import SimpleLogger, lstsq_init

logger = SimpleLogger()


def train_model(gs, train_loader, l2_penalty, lengthscale, args):
    num_inducing_points = args.num_inducing_points

    num_dims = sum(gs.num_dims) * 2 + 1
    model = RidgeGP(
        num_dims,
        lengthscale=lengthscale,
        num_inducing_points=num_inducing_points,
        intercept=gs.init_temp,
    )
    likelihood = GaussianLikelihood()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    likelihood = likelihood.to(device)

    lstsq_init(model, train_loader, device, l2_penalty)

    model.eval()
    all_coords = []
    all_residuals = []
    with torch.no_grad():
        for c, X, y_batch in train_loader:
            c, X, y_batch = (
                c.to(device),
                X.to(device),
                y_batch.to(device),
            )
            pred = model(c, X)
            all_coords.append(c.cpu())
            all_residuals.append((y_batch - pred.mean).cpu())

    all_coords = torch.cat(all_coords, dim=0)
    all_residuals = torch.cat(all_residuals, dim=0)

    k = min(num_inducing_points, len(all_coords))
    inducing_points = _farthest_point_sample(all_coords, all_residuals.abs(), k).to(
        device
    )
    model.gp_layer.variational_strategy.inducing_points.data.copy_(inducing_points)
    logger.info(f"Initialized {k} inducing points via farthest-point sampling.")

    dists = torch.cdist(inducing_points.cpu(), all_coords)
    nearest = dists.topk(k=min(10, len(all_coords)), dim=1, largest=False).indices
    variational_mean = all_residuals[nearest].mean(dim=1).to(device)
    model.gp_layer.variational_strategy._variational_distribution.variational_mean.data.copy_(
        variational_mean
    )

    pretrain_mse = (all_residuals**2).mean()
    logger.info(f"Pre-train MSE {pretrain_mse}")
    likelihood.noise_covar.noise = pretrain_mse.clamp(min=1e-4).to(device)

    residual_var = all_residuals.var()
    model.gp_layer.covar_module.kernels[0].outputscale = residual_var.clamp(
        min=1e-4
    ).to(device)
    model.gp_layer.covar_module.kernels[1].variance = (
        (residual_var * 0.1).clamp(min=1e-4).to(device)
    )

    logger.info(
        f"GP init — noise: {pretrain_mse.item():.4f}, "
        f"outputscale: {residual_var.item():.4f}"
    )

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=args.lr,
    )
    total_iters = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)
    mll = VariationalELBO(likelihood, model.gp_layer, num_data=len(all_coords))

    for i in range(args.epochs):
        model.train()
        likelihood.train()

        epoch_train_loss = 0
        epoch_train_count = 0
        for c, X, y_batch in train_loader:
            c, X, y_batch = (
                c.to(device),
                X.to(device),
                y_batch.to(device),
            )
            optimizer.zero_grad()
            output = model(c, X)
            loss = -mll(output, y_batch)
            loss += l2_penalty * model.beta.norm()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_train_loss += loss.item() * y_batch.size(0)
            epoch_train_count += y_batch.size(0)

        train_loss = epoch_train_loss / epoch_train_count
        logger.info(f"Epoch {i+1}/{args.epochs}, Train Loss: {train_loss:.4f}")

    model.eval()
    likelihood.eval()

    loss = 0
    with torch.no_grad():
        for c, X, y_batch in train_loader:
            c, X, y_batch = (
                c.to(device),
                X.to(device),
                y_batch.to(device),
            )
            output = model(c, X)
            loss -= mll(output, y_batch)

    return loss, model, likelihood


def extract_features(gs, indices, lengthscale):
    gs.precompute_features(lengthscale)
    return gs.get_all_features(np.asarray(indices))


def _farthest_point_sample(coords, residuals, k):
    n = coords.size(0)
    selected = [residuals.argmax().item()]
    min_dists = torch.full((n,), float("inf"))

    while len(selected) < k:
        last = coords[selected[-1]]
        dists = ((coords - last) ** 2).sum(dim=1)
        min_dists = torch.minimum(min_dists, dists)
        selected.append(min_dists.argmax().item())

    return coords[torch.tensor(selected)]


def _train_with_retry(gs, loader, l2_penalty, lengthscale, args, max_retries=3):
    jitter_levels = [1e-6, 1e-3, 1e-2]
    for attempt, jitter in enumerate(jitter_levels[:max_retries]):
        try:
            with gpytorch.settings.cholesky_jitter(jitter):
                return train_model(gs, loader, l2_penalty, lengthscale, args)
        except NotPSDError:
            if attempt < max_retries - 1:
                logger.warn(
                    f"Cholesky failed (jitter={jitter:.0e}), retrying with larger jitter…"
                )
            else:
                raise


def predict_causal_raster(gs, model, artifact, output_dir, args, tile_rows=256):
    """
    Tile the NLCD raster and write an 8-band causal effects GeoTIFF clipped to
    the spatial extent of the greenspace raster.

    For each GS class k, the causal effect at pixel p uses only the linear
    portion of RidgeGP, with the no-greenspace state as the implicit baseline
    (class-0 pixels contribute zero to all GS features):

        CE_k(p) = X_norm[11+k] * beta[11+k] + X_norm[30+k] * beta[30+k]

    The GP residual and intercept cancel vs the no-GS baseline.
    Indirect effect: weighted_GS features (indices 11–18).
    Direct effect:   point_GS features   (indices 30–37).
    """
    gs.precompute_features(artifact["lengthscale"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X_mean = artifact["X_mean"].to(device)
    X_std = artifact["X_std"].to(device)
    beta = model.beta.detach().to(device)

    n_nlcd = len(gs._NLCD_CLASSES)  # 11
    n_gs = len(gs._GREEN_CLASSES)  # 8
    weighted_gs_start = n_nlcd  # 11
    point_gs_start = n_nlcd + n_gs + n_nlcd  # 30

    # Compute the NLCD row/col window that covers the greenspace extent.
    H_full, W_full = gs.nlcd.data.shape
    gs_bounds_in_nlcd = transform_bounds(
        gs.greenspace.crs, gs.nlcd.crs, *gs.greenspace.bounds
    )
    nlcd_aff = AffineTransformer(gs.nlcd.transform)
    row0, col0 = nlcd_aff.rowcol(
        gs_bounds_in_nlcd[0], gs_bounds_in_nlcd[3]
    )  # left, top
    row1, col1 = nlcd_aff.rowcol(
        gs_bounds_in_nlcd[2], gs_bounds_in_nlcd[1]
    )  # right, bottom
    row_min = max(0, min(row0, row1))
    row_max = min(H_full, max(row0, row1))
    col_min = max(0, min(col0, col1))
    col_max = min(W_full, max(col0, col1))
    H_clip = row_max - row_min
    W_clip = col_max - col_min

    n_causal_bands = (
        n_gs  # one band per GS class; reference is implicit (no-GS = zero features)
    )
    causal_bands = np.full((n_causal_bands, H_clip, W_clip), np.nan, dtype=np.float32)

    for row_start in range(row_min, row_max, tile_rows):
        row_end = min(row_start + tile_rows, row_max)
        tile_H = row_end - row_start
        logger.info(f"Processing tile rows {row_start}–{row_end} / {row_max}")

        _coords, X = gs.get_raster_tile(row_start, row_end)
        X_norm = (X.to(device) - X_mean) / (X_std + 1e-8)

        with torch.no_grad():
            out_row = row_start - row_min
            for band_idx in range(n_gs):
                wi = weighted_gs_start + band_idx
                pi = point_gs_start + band_idx
                ce = X_norm[:, wi] * beta[wi] + X_norm[:, pi] * beta[pi]
                ce_full = ce.cpu().numpy().reshape(tile_H, W_full)
                causal_bands[band_idx, out_row : out_row + tile_H, :] = ce_full[
                    :, col_min:col_max
                ]

    out_path = os.path.join(output_dir, f"causal_effects_{args.time}.tif")
    clip_window = rasterio.windows.Window(col_min, row_min, W_clip, H_clip)
    with rasterio.open(gs.nlcd.data_dir) as ref:
        profile = ref.profile.copy()
        clip_transform = rasterio.windows.transform(clip_window, ref.transform)
    profile.update(
        count=n_causal_bands,
        dtype="float32",
        nodata=float("nan"),
        height=H_clip,
        width=W_clip,
        transform=clip_transform,
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        for band_idx in range(n_gs):
            gs_class = gs._GREEN_CLASSES[band_idx]
            dst.write(causal_bands[band_idx], band_idx + 1)
            dst.update_tags(
                band_idx + 1,
                description=f"Causal effect of GS class {gs_class} vs no-greenspace baseline (°C)",
            )

    logger.info(f"Causal effects raster saved → {out_path}")


def predict_temperature_raster(
    gs, model, likelihood, artifact, output_dir, args, tile_rows=256
):
    """
    Tile the NLCD raster (clipped to the greenspace extent) and write a 2-band
    predicted-temperature GeoTIFF.

      Band 1 — posterior predictive mean (°C)
      Band 2 — posterior predictive std  (°C)

    The predictive distribution comes from ``likelihood(model(c, X))``, which
    includes observation noise and gives a realistic uncertainty estimate.
    """
    gs.precompute_features(artifact["lengthscale"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    likelihood.eval()

    X_mean = artifact["X_mean"].to(device)
    X_std = artifact["X_std"].to(device)

    # Compute the same GS-extent clip window used by predict_causal_raster.
    H_full, W_full = gs.nlcd.data.shape
    gs_bounds_in_nlcd = transform_bounds(
        gs.greenspace.crs, gs.nlcd.crs, *gs.greenspace.bounds
    )
    nlcd_aff = AffineTransformer(gs.nlcd.transform)
    row0, col0 = nlcd_aff.rowcol(gs_bounds_in_nlcd[0], gs_bounds_in_nlcd[3])
    row1, col1 = nlcd_aff.rowcol(gs_bounds_in_nlcd[2], gs_bounds_in_nlcd[1])
    row_min = max(0, min(row0, row1))
    row_max = min(H_full, max(row0, row1))
    col_min = max(0, min(col0, col1))
    col_max = min(W_full, max(col0, col1))
    H_clip = row_max - row_min
    W_clip = col_max - col_min

    pred_mean = np.full((H_clip, W_clip), np.nan, dtype=np.float32)
    pred_std = np.full((H_clip, W_clip), np.nan, dtype=np.float32)

    for row_start in range(row_min, row_max, tile_rows):
        row_end = min(row_start + tile_rows, row_max)
        tile_H = row_end - row_start
        logger.info(
            f"Predicting temperature tile rows {row_start}–{row_end} / {row_max}"
        )

        coords, X = gs.get_raster_tile(row_start, row_end)
        X_norm = (X.to(device) - X_mean) / (X_std + 1e-8)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_pred = model(coords.to(device), X_norm)
            y_pred = likelihood(f_pred)
            mu = y_pred.mean.cpu().numpy()
            var = y_pred.variance.cpu().numpy()

        out_row = row_start - row_min
        pred_mean[out_row : out_row + tile_H, :] = mu.reshape(tile_H, W_full)[
            :, col_min:col_max
        ]
        pred_std[out_row : out_row + tile_H, :] = np.sqrt(np.maximum(var, 0.0)).reshape(
            tile_H, W_full
        )[:, col_min:col_max]

    out_path = os.path.join(output_dir, f"predicted_temperature_{args.time}.tif")
    clip_window = rasterio.windows.Window(col_min, row_min, W_clip, H_clip)
    with rasterio.open(gs.nlcd.data_dir) as ref:
        profile = ref.profile.copy()
        clip_transform = rasterio.windows.transform(clip_window, ref.transform)
    profile.update(
        count=2,
        dtype="float32",
        nodata=float("nan"),
        height=H_clip,
        width=W_clip,
        transform=clip_transform,
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pred_mean, 1)
        dst.write(pred_std, 2)
        dst.update_tags(1, description="Posterior predictive mean temperature (°C)")
        dst.update_tags(2, description="Posterior predictive std (°C)")

    logger.info(f"Predicted temperature raster saved → {out_path}")


def main(args):
    logger.info(args)

    if args.test:
        city = "asheville"
    else:
        task_id = os.getenv("SLURM_ARRAY_TASK_ID")
        cities = os.listdir(args.data_dir)
        city = cities[int(task_id)]

    logger.info(f"Running causal estimates for {city}...")
    output_dir = os.path.join(args.output_dir, city)
    os.makedirs(output_dir, exist_ok=True)

    try:
        gs = CausalGreenspaceDataset(
            os.path.join(args.data_dir, city),
            time=args.time,
            window_size=args.window_size,
            gs_downsample=args.gs_downsample,
        )
    except ValueError as e:
        logger.error(f"Problem opening dataset for {city}: {e}")
        return

    best_l2 = args.l2_penalty
    best_ls = args.lengthscale

    logger.info(f"Loaded hyperparams — l2: {best_l2:.4f}, lengthscale: {best_ls:.2f}")

    all_idx = np.arange(len(gs))
    coords, X, y = extract_features(gs, all_idx, best_ls)
    # pdb.set_trace()
    # All features are already in [0, 1] (GS/NLCD proportions, one-hot, and
    # elevation min-max scaled by the dataloader). No further normalization needed.
    ds = TensorDataset(coords, X, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    logger.info("Training model on all data...")
    _, model, likelihood = _train_with_retry(gs, loader, best_l2, best_ls, args)

    torch.save(model.state_dict(), os.path.join(output_dir, "causal_model.pth"))
    torch.save(
        likelihood.state_dict(), os.path.join(output_dir, "causal_likelihood.pth")
    )

    artifact = {
        "lengthscale": best_ls,
        "X_mean": torch.zeros(X.shape[1]),
        "X_std": torch.ones(X.shape[1]),
    }
    predict_causal_raster(gs, model, artifact, output_dir, args)
    predict_temperature_raster(gs, model, likelihood, artifact, output_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal Greenspace Effects")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing city traversal data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./causal_outputs",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--time",
        default="pm",
        type=str,
        choices=["am", "af", "pm"],
        help="Time of day for the data to use (am, af, pm).",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num-inducing-points",
        type=int,
        default=10,
        help="Number of GP inducing points.",
    )
    parser.add_argument(
        "--window-size",
        default=100,
        type=int,
        help="Spatial window radius in metres for feature extraction.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run on asheville only (bypasses SLURM task ID logic).",
    )
    parser.add_argument(
        "--l2-penalty",
        type=float,
        default=1.0,
        help="L2 penalty for the ridge regression initialization.",
    )
    parser.add_argument(
        "--lengthscale",
        type=float,
        default=100.0,
        help="Lengthscale for the Gaussian process kernel.",
    )
    parser.add_argument(
        "--gs-downsample",
        type=int,
        default=1,
        help="Downsample GS rasters by this factor to speed up training and reduce memory usage.",
    )
    arguments = parser.parse_args()
    main(arguments)
