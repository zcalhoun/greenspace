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
from torch.utils.data import TensorDataset, DataLoader, Subset
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO

sys.path.append("./src")
from dataloader_v2 import GreenspaceDataset
from model import CompleteModel, Exponential, RidgeGP
from utils import SimpleLogger, lstsq_init
from trainers import BayesianOptimization

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
    bo_variables = {"l2_penalty": (0.001, 1.0), "lengthscale": (50.0, 1000.0)}

    bayes_opt = BayesianOptimization(bo_variables, random_inits=10)

    train_dataset = Subset(gs, train_idx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    for bo_iter in range(args.bayes_opt_iters):
        trial = bayes_opt.get_next_trial()
        l2_penalty = trial.parameters["l2_penalty"]
        lengthscale = trial.parameters["lengthscale"]
        logger.info(f"BO - {trial.type} - l2: {l2_penalty} - ls: {lengthscale}")

        # Extract the features
        coords, X, y = extract_features(gs, train_loader, lengthscale)

        # Normalize X
        X_norm = normalize(X)
        sub_ds = TensorDataset(coords, X_norm, y)
        sub_loader = DataLoader(sub_ds, batch_size=args.batch_size, shuffle=True)
        mll, model, likelihood = train_model(
            gs, sub_loader, l2_penalty, lengthscale, args
        )
        trial.update(mll)

        # pdb.set_trace()
        logger.info(f"MLL: {mll:.2f}")
        logger.info(f"Noise: {likelihood.noise.detach().item():.2f}")
        logger.info(
            f"Model ls {model.gp_layer.covar_module.kernels[0].base_kernel.lengthscale.detach().item():.2f}"
        )


def normalize(X):
    mu = X.mean(axis=0)
    std = X.std(axis=0)

    X = (X - mu) / (std + 1e-8)

    return X


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

    # pdb.set_trace()
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

    # Likelihood noise: pre-training MSE is a direct estimate of unexplained variance
    pretrain_mse = (all_residuals**2).mean()
    logger.info(f"Pre-train MSE {pretrain_mse}")
    likelihood.noise_covar.noise = pretrain_mse.clamp(min=1e-4).to(device)

    residual_var = all_residuals.var()
    for kernel in model.gp_layer.covar_module.kernels:
        kernel.outputscale = residual_var.clamp(min=1e-4).to(device)

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


def extract_features(gs, data_loader, lengthscale):
    """
    Use the exponential function to extract the features from each of the covariates.
    """
    ds_params = zip(gs.window_size, gs.num_dims, gs.dimension_resolution)
    transforms = [
        Exponential(
            size=ws, num_dims=nd, dimension_resolution=dr, lengthscale=lengthscale
        )
        for ws, nd, dr in ds_params
    ]

    X = []
    coords = []
    y = []
    with torch.no_grad():
        for c, elev, stacked_window, temp in data_loader:
            combined_window = torch.column_stack(
                [t(s.flatten(start_dim=2)) for s, t in zip(stacked_window, transforms)]
            )

            point = torch.column_stack(
                [
                    s[:, :, ws // 2, ws // 2]
                    for ws, s in zip(gs.window_size, stacked_window)
                ]
            )

            X.extend(torch.column_stack([combined_window, point, elev]))
            coords.extend(c)
            y.extend(temp)

    return torch.stack(coords), torch.stack(X), torch.stack(y)


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


# def cross_validation(gs, train_idx, l2_penalty, args, folds=5):
#     """
#     This function creates a separate dataloader based on splitting the
#     training index, then trains the model on 5 folds, so we can get
#     a less noisy estimate of the model's performance.
#     """

#     training_errs = []
#     validation_errs = []
#     num_per_fold = len(train_idx) // folds
#     for i in range(folds):
#         logger.info(f"On CV fold {i} for L2 penalty {l2_penalty}...")
#         cv_train = train_idx.copy()
#         cv_val = np.concatenate(
#             [cv_train.pop(i * num_per_fold) for j in range(num_per_fold)]
#         )
#         cv_train = np.concatenate(cv_train)

#         train_ds = Subset(gs, cv_train)
#         val_ds = Subset(gs, cv_val)

#         # Use fewer inducing points and pretrain epochs during CV to save time
#         cv_result, _, _ = train(
#             train_ds,
#             val_ds,
#             l2_penalty,
#             args,
#             num_inducing_points=args.num_inducing_points,
#             pretrain_epochs=args.pretrain_epochs,
#         )
#         training_errs.append(cv_result["train_mse"])
#         validation_errs.append(cv_result["val_mse"])

#     return {
#         "l2_penalty": l2_penalty,
#         "train_mse_mean": np.mean(training_errs).item(),
#         "train_mse_std": np.std(training_errs).item(),
#         "val_mse_mean": np.mean(validation_errs).item(),
#         "val_mse_std": np.std(validation_errs).item(),
#     }


# def bopt_get_next_parameter(l2_lambdas, results):
#     X = np.log(np.array(l2_lambdas)).reshape(-1, 1)
#     y = np.array(results)

#     k = 0.5 * kernels.RBF()
#     gp = GaussianProcessRegressor(k, normalize_y=True)
#     gp.fit(X, y)

#     test_points = np.logspace(-4, 0, 1000)
#     test_points = np.log(test_points).reshape(-1, 1)

#     mu, std = gp.predict(test_points, return_std=True)

#     argmin = np.argmin(mu - std)
#     return np.exp(test_points[argmin][0])


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


def train(
    train_ds, val_ds, l2_penalty, args, num_inducing_points=None, pretrain_epochs=None
):
    """
    Fit a model and return the best outcome.
    """

    if num_inducing_points is None:
        num_inducing_points = args.num_inducing_points
    if pretrain_epochs is None:
        pretrain_epochs = args.pretrain_epochs

    model = CompleteModel(
        size=train_ds.dataset.window_size,
        num_dims=train_ds.dataset.num_dims,
        intercept=train_ds.dataset.init_temp,
        num_inducing_points=num_inducing_points,
        dimension_resolution=torch.tensor(train_ds.dataset.dimension_resolution),
    )
    likelihood = GaussianLikelihood()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = args.amp and use_cuda
    scaler = torch.amp.GradScaler(enabled=use_amp)
    model = model.to(device)
    likelihood = likelihood.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    training_result = {
        "l2_penalty": l2_penalty,
        "train_loss": [],
        "train_mse": None,
        "train_r2": None,
        "val_mse": None,
        "val_r2": None,
        "lengthscales": None,
        "coefficients": None,
    }

    ######
    ## LEAST SQUARES INITIALIZATION
    ######
    lstsq_init(model, train_loader, device, l2_penalty)
    logger.info("Initialized beta, elevation_weight, intercept via ridge regression.")

    ######
    ## PRE-TRAINING
    ######
    mse_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
    pretrain_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )
    for i in range(pretrain_epochs):
        model.train()
        likelihood.train()

        epoch_train_loss = 0
        epoch_train_count = 0
        for c, e, s, y_batch in train_loader:
            c, e, s, y_batch = (
                c.to(device),
                e.to(device),
                s.to(device),
                y_batch.to(device),
            )
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(c, e, s, pretrain=True)
                loss = mse_loss_fn(output, y_batch)
                loss += l2_penalty * model.beta.norm()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if torch.isnan(loss):
                logger.warn(f"NaN loss detected")
                logger.warn(f"  beta norm: {model.beta.norm().item()}")
                logger.warn(
                    f"  output range: {output.min().item():.3f} to {output.max().item():.3f}"
                )
                logger.warn(
                    f"  y_batch range: {y_batch.min().item():.3f} to {y_batch.max().item():.3f}"
                )
                logger.warn(f"  lengthscales: {model.exp_weight.lengthscale.data}")
                raise ValueError("NaN in pretrain loss — see above for diagnostics")
            epoch_train_loss += loss.item() * y_batch.size(0)
            epoch_train_count += y_batch.size(0)

        logger.info(
            f"Pretrain Epoch {i+1}/{pretrain_epochs}, Loss: {epoch_train_loss/epoch_train_count}"
        )
        train_loss = epoch_train_loss / epoch_train_count
        training_result["train_loss"].append(train_loss)
        pretrain_scheduler.step(train_loss)

    ######
    ## INDUCING POINT INITIALIZATION FROM PRE-TRAIN RESIDUALS
    ######
    model.eval()
    all_coords = []
    all_residuals = []
    with torch.no_grad():
        for c, e, s, y_batch in train_loader:
            c, e, s, y_batch = (
                c.to(device),
                e.to(device),
                s.to(device),
                y_batch.to(device),
            )
            pred = model(c, e, s, pretrain=True)
            all_coords.append(c.cpu())
            all_residuals.append((y_batch - pred).cpu())

    all_coords = torch.cat(all_coords, dim=0)  # (N, 2)
    all_residuals = torch.cat(all_residuals, dim=0)  # (N,) signed

    k = min(num_inducing_points, len(all_coords))
    inducing_points = _farthest_point_sample(all_coords, all_residuals.abs(), k).to(
        device
    )
    model.gp_layer.variational_strategy.inducing_points.data.copy_(inducing_points)
    logger.info(f"Initialized {k} inducing points via farthest-point sampling.")

    # Variational mean: for each inducing point, average the signed residuals
    # of its nearest training points — gives the GP a warm start on corrections
    # the linear model consistently misses at each location.
    dists = torch.cdist(inducing_points.cpu(), all_coords)  # (k, N)
    nearest = dists.topk(k=min(10, len(all_coords)), dim=1, largest=False).indices
    variational_mean = all_residuals[nearest].mean(dim=1).to(device)
    model.gp_layer.variational_strategy._variational_distribution.variational_mean.data.copy_(
        variational_mean
    )

    # Likelihood noise: pre-training MSE is a direct estimate of unexplained variance
    pretrain_mse = (all_residuals**2).mean()
    likelihood.noise_covar.noise = pretrain_mse.clamp(min=1e-4).to(device)

    # # Kernel outputscale: residual variance sets the GP's amplitude
    # residual_var = all_residuals.var()
    # for kernel in model.gp_layer.covar_module.kernels:
    #     kernel.outputscale = residual_var.clamp(min=1e-4).to(device)

    logger.info(
        f"GP init — noise: {pretrain_mse.item():.4f}, "
        f"outputscale: {residual_var.item():.4f}"
    )

    ######
    ## MAIN TRAINING LOOP
    ######
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": likelihood.parameters()},
        ],
        lr=args.lr,
    )
    total_iters = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)
    mll = VariationalELBO(likelihood, model.gp_layer, num_data=len(train_ds))
    best_train_loss = float("inf")
    patience_counter = 0

    for i in range(args.epochs):
        model.train()
        likelihood.train()

        epoch_train_loss = 0
        epoch_train_count = 0
        for c, e, s, y_batch in train_loader:
            c, e, s, y_batch = (
                c.to(device),
                e.to(device),
                s.to(device),
                y_batch.to(device),
            )
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(c, e, s)
                loss = -mll(output, y_batch)
                loss += l2_penalty * model.beta.norm()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(likelihood.parameters()),
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_train_loss += loss.item() * y_batch.size(0)
            epoch_train_count += y_batch.size(0)

        train_loss = epoch_train_loss / epoch_train_count
        training_result["train_loss"].append(train_loss)

        if train_loss < best_train_loss:
            if best_train_loss - train_loss < args.threshold:
                patience_counter += 1
            best_train_loss = train_loss
        else:
            patience_counter += 1

        logger.info(f"Epoch {i+1}/{args.epochs}, Train Loss: {train_loss:.4f}")
        if patience_counter >= args.patience:
            logger.info("Early stopping triggered.")
            break

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        train_mse = 0
        train_count = 0
        all_preds = []
        all_targets = []
        for c, e, s, y_batch in train_loader:
            c, e, s, y_batch = (
                c.to(device),
                e.to(device),
                s.to(device),
                y_batch.to(device),
            )

            pred = model(c, e, s)
            loss = mse_loss_fn(pred.mean, y_batch)
            train_mse += loss.item() * y_batch.size(0)
            train_count += y_batch.size(0)
            all_preds.extend(pred.mean.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

        # Calc r2 score
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2_score = 1 - ss_res / ss_tot
        training_result["train_r2"] = r2_score.item()
        train_mse /= train_count
        training_result["train_mse"] = train_mse

        if isinstance(val_ds, Subset):
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=use_cuda,
            )
            val_mse = 0
            val_count = 0
            all_preds = []
            all_targets = []
            for c, e, s, y_batch in val_loader:
                c, e, s, y_batch = (
                    c.to(device),
                    e.to(device),
                    s.to(device),
                    y_batch.to(device),
                )

                pred = model(c, e, s)
                loss = mse_loss_fn(pred.mean, y_batch)
                val_mse += loss.item() * y_batch.size(0)
                val_count += y_batch.size(0)
                all_preds.extend(pred.mean.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

            val_mse /= val_count
        else:
            logger.info("Performing cross-validation")
            fold_mses = []
            all_preds = []
            all_targets = []
            for fold in val_ds:
                val_loader = DataLoader(fold, batch_size=len(fold), shuffle=False)
                for c, e, s, y_batch in val_loader:
                    c, e, s, y_batch = (
                        c.to(device),
                        e.to(device),
                        s.to(device),
                        y_batch.to(device),
                    )

                    pred = likelihood(model(c, e, s))
                    loss = mse_loss_fn(pred.mean, y_batch)
                    fold_mses.append(loss.item())
                    all_preds.extend(pred.mean.cpu().numpy())
                    all_targets.extend(y_batch.cpu().numpy())

            val_mse = np.median(fold_mses)

        logger.info(f"Validation MSE: {val_mse}")
        # Calc r2 score
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2_score = 1 - ss_res / ss_tot
        training_result["val_r2"] = r2_score.item()
        training_result["val_mse"] = val_mse
        training_result["lengthscales"] = (
            model.exp_weight.lengthscale.detach().cpu().numpy().tolist()
        )
        training_result["coefficients"] = model.beta.detach().cpu().numpy().tolist()

    return training_result, model, likelihood


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
        "--pretrain-epochs",
        type=int,
        default=10,
        help="Number of epochs to pre-train the linear part of the model.",
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
        "--patience",
        default=5,
        type=int,
        help="Number of epochs to wait for improvement before early stopping.",
    )
    parser.add_argument(
        "--bayes-opt-iters",
        default=5,
        type=int,
        help="Number of iterations to perform for Bayesian optimization of the L2 penalty.",
    )
    parser.add_argument(
        "--pretrain-lr", default=0.01, type=float, help="Pre-training learning rate."
    )
    parser.add_argument(
        "--greenspace",
        action="store_true",
        help="Whether to include the greenspace window as part of the input features.",
    )
    parser.add_argument(
        "--threshold",
        default=0.001,
        type=float,
        help="Minimum improvement threshold for early stopping patience.",
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
        "--test", action="store_true", help="Whether to run in test mode."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--num-inducing-points",
        type=int,
        default=10,
        help="Number of GP inducing points for the final model.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision (CUDA only).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 to disable).",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Factor by which ReduceLROnPlateau reduces the pre-training learning rate.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=3,
        help="Epochs with no improvement before ReduceLROnPlateau reduces pre-training LR.",
    )
    parser.add_argument(
        "--cuda-debug",
        action="store_true",
        help=(
            "Set CUDA_LAUNCH_BLOCKING=1 to make all CUDA ops synchronous. "
            "Surfaces illegal memory access errors at the correct call site "
            "instead of a later sync point. Significantly slows training — "
            "use only for debugging."
        ),
    )
    arguments = parser.parse_args()
    main(arguments)
