"""
This notebook contains the code required to fit the Ridge+GP model, then produce
diagnostic results showing performance. As an additional step, the maps are then
produced using a FFT approach, so that we can capture the long-tailed effects
of vegetation on temperature.
"""

import os
import json
import argparse
import sys

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO

sys.path.append("./src")
from dataloader_v2 import GreenspaceDataset
from model import CompleteModel
from utils import SimpleLogger, lstsq_init, farthest_point_sample

# PANDAS NEEDS TO COME LAST...it's a weird dependency problem.
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the proper dataset for the experiment.
    try:
        gs = GreenspaceDataset(
            os.path.join(args.data_dir, city),
            greenspace=True,
            time=args.time,
            window_size=args.window_size,
            ndvi_albedo=False,
        )

    except ValueError as e:
        logger.error(f"Problem opening up dataset for {city}: {e}")
        return

    logger.info("Data loaded, proceeding to fit the model...")

    # Fit the model.
    model, likelihood, train_result, residuals = fit_model(gs, args)

    # Save the diagnostics in the proper place.
    with open(os.path.join(output_dir, "final_result.json"), "w") as f:
        json.dump(train_result, f)

    residuals.to_csv(os.path.join(output_dir, "residuals.csv"), index=False)

    # Save
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    torch.save(
        likelihood.state_dict(), os.path.join(output_dir, "final_likelihood.pth")
    )


def fit_model(dataset, args):

    num_inducing_points = args.num_inducing_points
    pretrain_epochs = args.pretrain_epochs
    l2_penalty = args.l2_penalty

    model = CompleteModel(
        size=dataset.window_size,
        num_dims=dataset.num_dims,
        intercept=dataset.init_temp,
        num_inducing_points=num_inducing_points,
        dimension_resolution=torch.tensor(dataset.dimension_resolution),
    )
    likelihood = GaussianLikelihood()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    scaler = torch.amp.GradScaler(enabled=False)
    model = model.to(device)
    likelihood = likelihood.to(device)

    if args.test:
        dataset = Subset(dataset, np.arange(0, len(dataset), 100))

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    training_result = {
        "train_loss": [],
        "train_mse": None,
        "train_r2": None,
        "lengthscales": None,
        "coefficients": None,
    }

    lstsq_init(model, train_loader, device, l2_penalty)
    logger.info("Least squares initialization complete")

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
            with torch.autocast(device_type=device.type, enabled=False):
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
    inducing_points = farthest_point_sample(all_coords, all_residuals.abs(), k).to(
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

    # Kernel outputscale: residual variance sets the GP's amplitude
    residual_var = all_residuals.var()
    for kernel in model.gp_layer.covar_module.kernels:
        kernel.outputscale = residual_var.clamp(min=1e-4).to(device)

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
    mll = VariationalELBO(likelihood, model.gp_layer, num_data=len(dataset))
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
            with torch.autocast(device_type=device.type, enabled=False):
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
        all_coords = []
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
            all_coords.extend(c.cpu())

        # Calc r2 score
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_coords = np.array(all_coords)
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2_score = 1 - ss_res / ss_tot
        training_result["train_r2"] = r2_score.item()
        train_mse /= train_count
        training_result["train_mse"] = train_mse

        training_result["lengthscales"] = (
            model.exp_weight.lengthscale.detach().cpu().numpy().tolist()
        )
        training_result["coefficients"] = model.beta.detach().cpu().numpy().tolist()

        residuals = pd.DataFrame(
            {
                "x": all_coords[:, 0],
                "y": all_coords[:, 1],
                "preds": all_preds,
                "targets": all_targets,
                "residuals": all_targets - all_preds,
            }
        )

        return model, likelihood, training_result, residuals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greenspace Map Creation.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/ash21/asheville/pm/",
        help="Directory containing the data.",
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
        "--pretrain-lr", default=0.01, type=float, help="Pre-training learning rate."
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
        default=51,
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
        "--l2-penalty",
        type=float,
        default=0.01,
        help="A default L2 penalty for training the model.",
    )
    arguments = parser.parse_args()
    main(arguments)
