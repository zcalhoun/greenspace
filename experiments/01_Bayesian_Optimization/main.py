import os
import json
import argparse
import sys
import pdb

# Must be set before any CUDA-initializing imports.
# Activated by --cuda-debug to make all CUDA ops synchronous so errors surface
# at the correct call site rather than at the next CPU-GPU sync point.
if "--cuda-debug" in sys.argv:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np

# from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from linear_operator.utils.errors import NotPSDError

from greenspace.dataloader import GreenspaceDataset
from greenspace.model import CompleteModel, RidgeGP
from greenspace.utils import SimpleLogger, lstsq_init
from greenspace.trainers import BayesianOptimization

import pandas as pd

logger = SimpleLogger()


def main(args):

    logger.info(args)

    if args.test:
        city = "asheville"
    else:
        task_id = os.getenv("SLURM_ARRAY_TASK_ID")
        cities = os.listdir(args.data_dir)
        city = cities[int(task_id)]

    logger.info(f"Running code for {city}...")
    output_dir = os.path.join(args.output_dir, city)
    retry = False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        retry = True

    try:
        gs = GreenspaceDataset(
            os.path.join(args.data_dir, city),
            greenspace=args.greenspace,
            time=args.time,
            window_size=args.window_size,
            gs_downsample=args.gs_downsample,
        )
    except ValueError as e:
        logger.error(f"Problem opening up dataset for {city}: {e}")
        return

    logger.info("Performing block split of the data...")
    train_idx, test_idx = block_split(gs, args.num_clusters)

    logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    ######
    # Initialize the training set up.
    ######
    bo_variables = {
        "l2_penalty": (args.l2_min, args.l2_max),
        "lengthscale": (50.0, 500.0),
    }

    bayes_opt = BayesianOptimization(bo_variables, random_inits=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trial_artifacts = []
    for bo_iter in range(args.bayes_opt_iters):
        trial = bayes_opt.get_next_trial()
        l2_penalty = trial.parameters["l2_penalty"]
        lengthscale = trial.parameters["lengthscale"]
        logger.info(
            f"BO iter {bo_iter} - {trial.type} | ls: {lengthscale:.2f}, l2: {l2_penalty:.4f}"
        )

        # Train on the training split.
        coords, X, y = extract_features(gs, train_idx, lengthscale)
        sub_ds = TensorDataset(coords, X, y)
        sub_loader = DataLoader(sub_ds, batch_size=args.batch_size, shuffle=True)
        try:
            _, model, likelihood = _train_with_retry(
                gs, sub_loader, l2_penalty, lengthscale, args
            )
        except NotPSDError:
            logger.warn(
                f"BO iter {bo_iter}: Cholesky failed on all retries, skipping trial."
            )
            trial.skip()
            trial_artifacts.append(None)
            save_result(
                {
                    "bo_iter": bo_iter,
                    "trial_type": trial.type,
                    "test_mse": None,
                    "l2_penalty": float(l2_penalty),
                    "lengthscale": float(lengthscale),
                    "skipped": True,
                },
                output_dir,
            )
            continue

        # Evaluate on the held-out test split — this is the BO objective.
        # test_idx shares the same lengthscale precomputation as train_idx above.
        test_coords, test_X, test_y = extract_features(gs, test_idx, lengthscale)
        test_loader_bo = DataLoader(
            TensorDataset(test_coords, test_X, test_y),
            batch_size=args.batch_size,
            shuffle=False,
        )
        model.eval()
        likelihood.eval()
        bo_preds = []
        with torch.no_grad():
            for c, Xb, _ in test_loader_bo:
                bo_preds.append(model(c.to(device), Xb.to(device)).mean.cpu())
        bo_preds = torch.cat(bo_preds)
        test_mse = ((bo_preds - test_y) ** 2).mean().item()
        ss_tot = ((test_y - test_y.mean()) ** 2).sum().item()
        test_r2 = (
            (1 - ((bo_preds - test_y) ** 2).sum().item() / ss_tot)
            if ss_tot > 0
            else float("nan")
        )

        trial.update(test_mse)

        logger.info(f"Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}")
        logger.info(f"Noise: {likelihood.noise.detach().item():.4f}")

        trial_artifacts.append(
            {
                "test_mse": test_mse,
                "test_r2": test_r2,
                "l2_penalty": float(l2_penalty),
                "lengthscale": float(lengthscale),
                "X_mean": torch.zeros(X.shape[1]),
                "X_std": torch.ones(X.shape[1]),
                "model": model,
                "likelihood": likelihood,
            }
        )
        save_result(
            {
                "bo_iter": bo_iter,
                "trial_type": trial.type,
                "test_mse": test_mse,
                "test_r2": test_r2,
                "l2_penalty": float(l2_penalty),
                "lengthscale": float(lengthscale),
            },
            output_dir,
        )

    best_trial = bayes_opt.get_best_trial()
    best_idx = bayes_opt.trials.index(best_trial)
    best_artifact = trial_artifacts[best_idx]
    best_model = best_artifact["model"]
    best_likelihood = best_artifact["likelihood"]

    torch.save(best_model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    torch.save(
        best_likelihood.state_dict(), os.path.join(output_dir, "final_likelihood.pth")
    )

    final_result = {
        "best_bo_iter": best_idx,
        "best_l2_penalty": best_artifact["l2_penalty"],
        "best_lengthscale": best_artifact["lengthscale"],
        "best_test_mse": best_artifact["test_mse"],
        "best_test_r2": best_artifact["test_r2"],
        "X_mean": best_artifact["X_mean"].tolist(),
        "X_std": best_artifact["X_std"].tolist(),
    }
    with open(os.path.join(output_dir, "final_result.json"), "w") as f:
        json.dump(final_result, f, indent=2)

    logger.info(
        f"Best trial (iter {best_idx}): l2={best_artifact['l2_penalty']:.4f}, "
        f"ls={best_artifact['lengthscale']:.1f}, "
        f"Test MSE={best_artifact['test_mse']:.4f}, Test R²={best_artifact['test_r2']:.4f}"
    )

    # Refit on all data (train + test) using the best hyperparameters.
    # The test metrics above are an honest estimate of generalisation error;
    # now use every available observation to get the best possible spatial model.
    logger.info("Refitting final model on all data (train + test) …")
    all_idx = np.concatenate([train_idx, test_idx])
    coords_all, X_all, y_all = extract_features(
        gs, all_idx, best_artifact["lengthscale"]
    )
    all_ds = TensorDataset(coords_all, X_all, y_all)
    all_loader = DataLoader(all_ds, batch_size=args.batch_size, shuffle=True)
    _, final_model, final_likelihood = _train_with_retry(
        gs, all_loader, best_artifact["l2_penalty"], best_artifact["lengthscale"], args
    )

    # Overwrite the saved weights with the all-data model.
    torch.save(final_model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    torch.save(
        final_likelihood.state_dict(),
        os.path.join(output_dir, "final_likelihood.pth"),
    )

    logger.info("Generating full-raster temperature prediction …")
    predict_raster(gs, final_model, final_likelihood, best_artifact, output_dir, args)


def train_model(gs, train_loader, l2_penalty, lengthscale, args):
    num_inducing_points = args.num_inducing_points

    # Set up mode
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

    # Initialize with least squares
    lstsq_init(model, train_loader, device, l2_penalty)
    # pdb.set_trace()
    # Initialize inducing points
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

    all_coords = torch.cat(all_coords, dim=0)  # (N, 2)
    all_residuals = torch.cat(all_residuals, dim=0)  # (N,) signed

    k = min(num_inducing_points, len(all_coords))
    inducing_points = _farthest_point_sample(all_coords, all_residuals.abs(), k).to(
        device
    )
    model.gp_layer.variational_strategy.inducing_points.data.copy_(inducing_points)
    logger.info(f"Initialized {k} inducing points via farthest-point sampling.")

    dists = torch.cdist(inducing_points.cpu(), all_coords)  # (k, N)
    nearest = dists.topk(k=min(10, len(all_coords)), dim=1, largest=False).indices
    variational_mean = all_residuals[nearest].mean(dim=1).to(device)
    model.gp_layer.variational_strategy._variational_distribution.variational_mean.data.copy_(
        variational_mean
    )

    # Likelihood noise: pre-training MSE is a direct estimate of unexplained variance.
    # GPyTorch stores noise as variance, so assign MSE directly (not sqrt).
    pretrain_mse = (all_residuals**2).mean()
    logger.info(f"Pre-train MSE {pretrain_mse}")
    likelihood.noise_covar.noise = pretrain_mse.clamp(min=1e-4).to(device)

    # ScaleKernel uses .outputscale; LinearKernel uses .variance — they are different
    # GPyTorch attributes and must be set separately.
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
    # Set up optimizer
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
    best_train_loss = float("inf")

    # Train for 10 epochs to fine-tune.
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
        # training_result["train_loss"].append(train_loss)

        logger.info(f"Epoch {i+1}/{args.epochs}, Train Loss: {train_loss:.4f}")

    # Calculate -MLL
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

    # Need to return -MLL as main-result, but we should monitor the
    # terms of the GP, too, so we can observe performance.
    return loss, model, likelihood


def extract_features(gs, indices, lengthscale):
    """
    Precompute full-raster exponentially-weighted features via FFT convolution,
    then perform vectorised point lookups for the requested sample indices.

    The convolution result is cached inside ``gs``; repeated calls with the
    same lengthscale pay only the lookup cost.

    Args:
        gs:          GreenspaceDataset instance
        indices:     array-like of integer sample indices
        lengthscale: kernel decay lengthscale in metres

    Returns:
        coords: (N, 2) float tensor
        X:      (N, D) float tensor of features
        y:      (N,)  float tensor of temperatures
    """
    gs.precompute_features(lengthscale)
    return gs.get_all_features(np.asarray(indices))


def get_last_result(output_dir):
    try:
        output_path = os.path.join(output_dir, "training_results.jsonl")
        df = pd.read_json(output_path, lines=True)

        results = df["val_mse_mean"].values.tolist()
        l2_lambdas = df["l2_penalty"].values.tolist()

        return results, l2_lambdas
    except Exception as e:
        print(e)
        logger.info(f"No files found")
        return [], []


def save_result(train_result, output_dir):
    """Save the training result to a JSONL file."""
    output_path = os.path.join(output_dir, "training_results.jsonl")
    with open(output_path, "a") as f:
        json.dump(train_result, f)
        f.write("\n")


def _farthest_point_sample(coords, residuals, k):
    """
    Select k points from coords using farthest-point sampling seeded by the
    highest residual. Each subsequent point maximises the minimum distance to
    the already-selected set, so the result is both spatially spread and
    biased toward high-error regions.

    Args:
        coords:    (N, 2) float tensor of standardized coordinates
        residuals: (N,)   float tensor of absolute residuals
        k:         number of points to select

    Returns:
        (k, 2) float tensor of selected coordinates
    """
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


def predict_raster(
    gs, model, likelihood, best_artifact, output_dir, args, tile_rows=256
):
    """
    Tile the NLCD raster and predict temperature at every pixel.

    Writes ``<output_dir>/prediction.tif``: a two-band float32 GeoTIFF in the
    NLCD raster's CRS and extent.
      Band 1 — posterior predictive mean (°C)
      Band 2 — posterior predictive std  (°C)

    The predictive distribution (from ``likelihood(model(c, X))``) includes
    observation noise, giving a realistic uncertainty estimate.

    Args:
        gs:            GreenspaceDataset (precompute_features already called for
                       best_artifact["lengthscale"])
        model:         trained RidgeGP
        likelihood:    trained GaussianLikelihood
        best_artifact: dict with keys "lengthscale", "X_mean", "X_std"
        output_dir:    directory where prediction.tif is written
        args:          parsed CLI args (used for batch_size)
        tile_rows:     number of NLCD rows to process per tile
    """
    import rasterio

    gs.precompute_features(best_artifact["lengthscale"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    likelihood.eval()

    X_mean = best_artifact["X_mean"]
    X_std = best_artifact["X_std"]

    H, W = gs.nlcd.data.shape
    pred_mean = np.full((H, W), np.nan, dtype=np.float32)
    pred_std = np.full((H, W), np.nan, dtype=np.float32)

    for row_start in range(0, H, tile_rows):
        row_end = min(row_start + tile_rows, H)
        logger.info(f"Predicting tile rows {row_start}–{row_end} / {H}")

        coords, X = gs.get_raster_tile(row_start, row_end)
        X_norm = (X - X_mean) / (X_std + 1e-8)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_pred = model(coords.to(device), X_norm.to(device))
            y_pred = likelihood(f_pred)
            mu = y_pred.mean.cpu().numpy()
            var = y_pred.variance.cpu().numpy()

        tile_H = row_end - row_start
        pred_mean[row_start:row_end, :] = mu.reshape(tile_H, W)
        pred_std[row_start:row_end, :] = np.sqrt(np.maximum(var, 0.0)).reshape(
            tile_H, W
        )

    out_path = os.path.join(output_dir, "prediction.tif")
    with rasterio.open(gs.nlcd.data_dir) as ref:
        profile = ref.profile.copy()
    profile.update(count=2, dtype="float32", nodata=float("nan"))

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pred_mean, 1)
        dst.write(pred_std, 2)
        dst.update_tags(1, description="Posterior predictive mean temperature (°C)")
        dst.update_tags(2, description="Posterior predictive std (°C)")

    logger.info(f"Prediction raster saved → {out_path}")


def block_split(gs, num_clusters):
    """
    This function applies KMeans to the coordinates to split the data
    into a train and test dataset.

    The training dataset is an array of indices in each cluster, which is then
    used for performing 5-fold cross-validation.

    The training dataset is just the array of indices.
    """

    # Get coordinates
    ac = gs.coords

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(ac)

    clusters = np.arange(num_clusters)
    np.random.seed(0)
    np.random.shuffle(clusters)

    indices = np.arange(len(gs))

    training_size = int(num_clusters * 0.8)

    train_idx = indices[np.isin(kmeans.labels_, clusters[:training_size])]
    test_idx = indices[np.isin(kmeans.labels_, clusters[training_size:])]

    return train_idx, test_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greenspace Analysis")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/ash21/asheville/pm/",
        help="Directory containing the data.",
    )
    parser.add_argument(
        "--num-clusters",
        default=50,
        type=int,
        help="The number of clusters to use for KMeans.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../outputs",
        help="Directory to save the output plots and results.",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="The number of epochs to use for the main training loop.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="The batch size number to use for training.",
    )
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
        help="Learning rate for the main training loop.",
    )
    parser.add_argument(
        "--bayes-opt-iters",
        default=50,
        type=int,
        help="Number of iterations to perform for Bayesian optimization of the L2 penalty.",
    )
    parser.add_argument(
        "--greenspace",
        action="store_true",
        help="Whether to include the greenspace window as part of the input features.",
    )
    # Add time to be one of ["am", "af", "pm"]
    parser.add_argument(
        "--time",
        default="pm",
        type=str,
        choices=["am", "af", "pm"],
        help="Time of day for the data to use (am, af, pm).",
    )
    parser.add_argument(
        "--window-size",
        default=100,
        type=int,
        help="The size of the window to use for the input features.",
    )
    parser.add_argument(
        "--gs-downsample",
        type=int,
        default=1,
        help=(
            "Spatial downsampling factor applied to the greenspace window before "
            "the Exponential transform. The greenspace raster is 0.6 m/pixel, so "
            "sub-pixel detail is wasted; a factor of 5 reduces it to 3 m/pixel and "
            "shrinks the intermediate tensor area by 25×. Use 1 (default) to disable."
        ),
    )
    parser.add_argument(
        "--test", action="store_true", help="Whether to run in test mode."
    )
    parser.add_argument(
        "--num-inducing-points",
        type=int,
        default=10,
        help="Number of GP inducing points for the final model.",
    )
    parser.add_argument(
        "--l2-min",
        type=float,
        default=0.0001,
        help="Lower bound of the L2 penalty search range for Bayesian optimization.",
    )
    parser.add_argument(
        "--l2-max",
        type=float,
        default=0.1,
        help="Upper bound of the L2 penalty search range for Bayesian optimization.",
    )
    arguments = parser.parse_args()
    main(arguments)
