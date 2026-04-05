import os
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


def main(args):

    gs = GreenspaceDataset(args.data_dir, greenspace=args.greenspace)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_idx, val_idx, test_idx = block_split(gs, args.num_clusters, args.output_dir)

    train_dataset = Subset(gs, train_idx)
    val_dataset = Subset(gs, val_idx)
    test_dataset = Subset(gs, test_idx)

    ######
    # Initialize the training set up.
    ######
    l2_lambdas = [1e-2, 1e-1, 1, 10, 100]

    results = []
    for l2_penalty in l2_lambdas:
        train_result, model, likelihood = train(
            train_dataset, val_dataset, l2_penalty, args
        )
        results.append(train_result)

    for i in range(args.bayes_opt_iters):
        # Get the next set of hyperparameters to try
        print("Bayesian optimization iteration", i + 1)

        l2_penalty = bopt_get_next_parameter(l2_lambdas, results)

        # Train the model with these hyperparameters
        train_result, model, likelihood = train(
            train_dataset, val_dataset, l2_penalty, args
        )
        results.append(train_result)

    #####
    # Final model training with the best hyperparameters
    #####
    best_l2_penalty = l2_lambdas[np.argmin(results)]
    print("Best L2 penalty found:", best_l2_penalty)
    # combine training and validation datasets for final training
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

    test_error, model, likelihood = train(
        combined_dataset, test_dataset, best_l2_penalty, args
    )
    print("Test MSE with best L2 penalty:", test_error)

    error = {"test_mse": test_error, "best_l2_penalty": best_l2_penalty}
    np.save(os.path.join(args.output_dir, "results.npy"), error)
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

    test_points = np.logspace(-2, 2, 100)
    test_points = np.log(test_points).reshape(-1, 1)

    mu, std = gp.predict(test_points, return_std=True)

    argmax = np.argmax(mu + std)
    return np.exp(test_points[argmax][0])


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
        print(
            f"Pretrain Epoch {i+1}/{args.pretrain_epochs}, Loss: {epoch_train_loss/epoch_train_count}"
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

        if train_loss < best_train_loss:
            if best_train_loss - train_loss < 0.01:
                patience_counter += 1
            best_train_loss = train_loss
        else:
            patience_counter += 1

        print(f"Epoch {i+1}/{args.epochs}, Train Loss: {train_loss:.4f}")
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    model.eval()
    likelihood.eval()
    epoch_val_loss = 0
    epoch_val_count = 0

    with torch.no_grad():
        for c, e, s, y_batch in val_loader:
            c, e, s, y_batch = (
                c.to(device),
                e.to(device),
                s.to(device),
                y_batch.to(device),
            )

            pred = model(c, e, s)
            loss = mse_loss_fn(pred.mean, y_batch)
            epoch_val_loss += loss.item() * y_batch.size(0)
            epoch_val_count += y_batch.size(0)

    val_loss = epoch_val_loss / epoch_val_count

    return val_loss, model, likelihood


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
    arguments = parser.parse_args()
    main(arguments)
