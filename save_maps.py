"""
This notebook takes the learned model and creates TIFF files from the results.

"""

import os
import json
import argparse
import sys

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO

sys.path.append("./src")
from dataloader_v2 import GreenspaceDataset, _NLCD_CLASSES, _GREEN_CLASSES
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

    # Load the proper dataset for the experiment.
    gs = GreenspaceDataset(
        os.path.join(args.data_dir, city),
        greenspace=True,
        time=args.time,
        window_size=51,  # THIS DOESN'T MATTER.
        ndvi_albedo=False,
        non_negative=True,
        return_elev_tif=True,
    )

    # Load the model parameters
    if args.test:
        size = 51
        num_ip = 10
    else:
        size = 201
        num_ip = 100

    model = CompleteModel(
        size=size,
        num_dims=28,
        intercept=gs.init_temp,
        num_inducing_points=num_ip,
        dimension_resolution=torch.tensor(gs.dimension_resolution),
        non_negative=True,
    )
    state_dict = torch.load(os.path.join(output_dir, "final_model.pth"))
    model.load_state_dict(state_dict)
    logger.info("Model loaded")

    # Pre-process the datasets using the Fourier transform approach
    lengthscale = model.exp_weight.lengthscale.detach().numpy()
    logger.info(f"LS: {lengthscale}")

    ###################################################
    # Reproject elevation, NLCD to match Greenspace bounds
    ###################################################

    traversal = gpd.read_file(os.path.join(args.data_dir, city, args.time, "trav.shp"))

    with rasterio.open(os.path.join(args.data_dir, city, "greenspace.tif")) as src:
        greenspace = src.data(1)

    with rasterio.open(os.path.join(args.data_dir, city, "nlcd.tif")) as src:
        nlcd = src.data(1)

    with rasterio.open(os.path.join(args.data_dir, city, "elevation.tif")) as src:
        elevation = src.data(1)

    # TODO:
    #   First, clip the greenspace, nlcd, and elevation data based on the traversal bounds + a 500 meter buffer
    #   Second, reproject the data to be at the greenspace resolution.
    #   Third, create one-hot encodings for the NLCD and Greenspace classes from these data.
    #   Four, take the Fourier transform of the one-hot encodings and multiply it by the Fourier transform
    #       of the exponential function, based on the learned lengthscale from the model.


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
        "--test", action="store_true", help="Whether to run in test mode."
    )
    parser.add_argument(
        "--time",
        default="pm",
        type=str,
        choices=["am", "af", "pm"],
        help="Time of day for the data to use (am, af, pm).",
    )

    arguments = parser.parse_args()
    main(arguments)
