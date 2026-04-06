import os
import json
import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO

sys.path.append("./src")
from dataloader import GreenspaceDataset
from model import CompleteModel
from utils import SimpleLogger

logger = SimpleLogger()


def main(args):

    logger.info(args)

    gs = GreenspaceDataset(args.data_dir, greenspace=args.greenspace)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Performing block split of the data...")
    train_idx, val_idx, test_idx = block_split(gs, args.num_clusters, args.output_dir)

    train_dataset = Subset(gs, train_idx)
    val_dataset = Subset(gs, val_idx)
    test_dataset = Subset(gs, test_idx)

    ######
    # Initialize the training set up.
    ######
    l2_lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    results = []
    for l2_penalty in l2_lambdas:
        logger.info(f"Training with L2 penalty: {l2_penalty}")
        train_result, model, likelihood = train(
            train_dataset, val_dataset, l2_penalty, args
        )
        logger.info(f"Validation MSE: {train_result['val_mse']}")
        results.append(train_result["val_mse"])
        save_result(train_result, args.output_dir)

    for i in range(args.bayes_opt_iters):
        # Get the next set of hyperparameters to try
        logger.info(f"Bayesian optimization iteration {i + 1}")

        l2_penalty = bopt_get_next_parameter(l2_lambdas, results)
        logger.info(f"Next L2 penalty to try: {l2_penalty}")
        # Train the model with these hyperparameters
        train_result, model, likelihood = train(
            train_dataset, val_dataset, l2_penalty, args
        )
        logger.info(f"Validation MSE: {train_result['val_mse']}")
        l2_lambdas.append(l2_penalty)
        results.append(train_result["val_mse"])
        save_result(train_result, args.output_dir)

    #####
    # Final model training with the best hyperparameters
    #####
    best_l2_penalty = l2_lambdas[np.argmin(results)]
    logger.info(f"Best L2 penalty found: {best_l2_penalty}")
    # combine training and validation datasets for final training

    full_train_idx = np.concatenate([train_idx, val_idx])
    full_train_dataset = Subset(gs, full_train_idx)

    train_result, model, likelihood = train(
        full_train_dataset, test_dataset, best_l2_penalty, args
    )
    logger.info(f"Test MSE with best L2 penalty: {train_result['val_mse']}")

    with open(os.path.join(args.output_dir, "final_result.json"), "w") as f:
        json.dump(train_result, f)

    # Save
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    torch.save(
        likelihood.state_dict(), os.path.join(args.output_dir, "final_likelihood.pth")
    )


def bopt_get_next_parameter(l2_lambdas, results):
    X = np.log(np.array(l2_lambdas)).reshape(-1, 1)
    y = np.array(results)

    k = 0.5 * kernels.RBF()
    gp = GaussianProcessRegressor(k)
    gp.fit(X, y)

    test_points = np.logspace(-6, 0, 1000)
    test_points = np.log(test_points).reshape(-1, 1)

    mu, std = gp.predict(test_points, return_std=True)

    argmin = np.argmin(mu - std)
    return np.exp(test_points[argmin][0])


def save_result(train_result, output_dir):
    """Save the training result to a JSONL file."""
    output_path = os.path.join(output_dir, "training_results.jsonl")
    with open(output_path, "a") as f:
        json.dump(train_result, f)
        f.write("\n")


def train(train_ds, val_ds, l2_penalty, args):
    """
    Fit a model and return the best outcome.
    """

    model = CompleteModel(
        size=train_ds.dataset.window_size,
        num_dims=train_ds.dataset.num_dims,
        intercept=train_ds.dataset.init_temp,
    )
    likelihood = GaussianLikelihood()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    likelihood = likelihood.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

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
    ## PRE-TRAINING
    ######
    mse_loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)
    for i in range(args.pretrain_epochs):
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
            output = model(c, e, s, pretrain=True)
            loss = mse_loss_fn(output, y_batch)
            loss += l2_penalty * model.beta.norm()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * y_batch.size(0)
            epoch_train_count += y_batch.size(0)
        logger.info(
            f"Pretrain Epoch {i+1}/{args.pretrain_epochs}, Loss: {epoch_train_loss/epoch_train_count}"
        )
        training_result["train_loss"].append(epoch_train_loss / epoch_train_count)

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
            output = model(c, e, s)
            loss = -mll(output, y_batch)
            loss += l2_penalty * model.beta.norm()
            loss.backward()
            optimizer.step()
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

        # Calc r2 score
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2_score = 1 - ss_res / ss_tot
        training_result["val_r2"] = r2_score.item()

        val_mse /= val_count
        training_result["val_mse"] = val_mse
        training_result["lengthscales"] = (
            model.exp_weight.lengthscale.detach().cpu().numpy().tolist()
        )
        training_result["coefficients"] = model.beta.detach().cpu().numpy().tolist()

    return training_result, model, likelihood


def block_split(gs, num_clusters, output_dir=None):

    # Get coordinates
    ac = []
    for i in range(len(gs)):
        c, _, _, _ = gs[i]
        ac.append(c)
    ac = np.stack([c.numpy() for c in ac])

    # Number of clusters
    k = 50
    # Fit KMeans

    kmeans = KMeans(n_clusters=k, random_state=0).fit(ac)

    # Randomly sample 30 clusters

    clusters = np.arange(50)
    np.random.seed(0)
    np.random.shuffle(clusters)

    indices = np.arange(len(gs))

    train_idx = indices[np.isin(kmeans.labels_, clusters[:30])]
    val_idx = indices[np.isin(kmeans.labels_, clusters[30:40])]
    test_idx = indices[np.isin(kmeans.labels_, clusters[40:])]

    if output_dir is not None:
        # Create a plot showing the clusters and the train/val/test splits
        plt.scatter(ac[train_idx, 0], ac[train_idx, 1], label="Train", alpha=0.5)
        plt.scatter(ac[val_idx, 0], ac[val_idx, 1], label="Val", alpha=0.5)
        plt.scatter(ac[test_idx, 0], ac[test_idx, 1], label="Test", alpha=0.5)
        plt.legend()
        plt.title("KMeans Clusters and Train/Val/Test Splits")
        plt.savefig(os.path.join(output_dir, "kmeans_clusters.png"))
    return train_idx, val_idx, test_idx


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
        default=64,
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
    arguments = parser.parse_args()
    main(arguments)
