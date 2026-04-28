"""
This notebook takes the learned model and creates TIFF files from the results.

"""

import os
import json
import argparse
import sys

import rasterio
import rasterio.windows
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import geopandas as gpd
from pyproj import Transformer, Geod

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

    lengthscale = model.exp_weight.lengthscale.item()
    logger.info(f"LS: {lengthscale:.2f} m")
    beta = model.beta.detach().numpy()  # (num_dims * 2,) = (56,)
    elevation_weight = model.elevation_weight.item()  # scalar < 0
    intercept = model.intercept.item()

    ###################################################
    # Step 1: Clip rasters to traversal bounds + 500 m buffer
    ###################################################

    traversal = gpd.read_file(os.path.join(args.data_dir, city, args.time, "trav.shp"))
    trav_metric = traversal.to_crs("EPSG:5070")
    bounds = trav_metric.total_bounds  # [xmin, ymin, xmax, ymax] in metres
    buff = bounds + np.array([-500.0, -500.0, 500.0, 500.0])

    city_dir = os.path.join(args.data_dir, city)

    # Clip the greenspace TIF to the buffered traversal bounds and use it as
    # the reference grid (CRS, transform, and pixel size) for all other rasters.
    with rasterio.open(os.path.join(city_dir, "greenspace.tif")) as src:
        to_gs = Transformer.from_crs("EPSG:5070", src.crs, always_xy=True)
        xs, ys = to_gs.transform([buff[0], buff[2]], [buff[1], buff[3]])
        win_raw = rasterio.windows.from_bounds(
            min(xs), min(ys), max(xs), max(ys), src.transform
        )
        # Clamp to the raster's actual extent so the window offsets stay
        # non-negative and window_transform gives the correct origin.
        col_off = max(0, int(np.floor(win_raw.col_off)))
        row_off = max(0, int(np.floor(win_raw.row_off)))
        col_end = min(src.width, int(np.ceil(win_raw.col_off + win_raw.width)))
        row_end = min(src.height, int(np.ceil(win_raw.row_off + win_raw.height)))
        win = rasterio.windows.Window(
            col_off, row_off, col_end - col_off, row_end - row_off
        )
        gs_data = src.read(1, window=win)
        ref_transform = src.window_transform(win)
        ref_crs = src.crs

    H, W = gs_data.shape
    logger.info(f"Map grid: {H} x {W}")

    ###################################################
    # Step 2: Reproject NLCD and elevation to the greenspace reference grid
    ###################################################

    nlcd_data = np.zeros((H, W), dtype=np.uint8)
    with rasterio.open(os.path.join(city_dir, "nlcd.tif")) as src:
        reproject(
            source=src.read(1),
            destination=nlcd_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
        )

    elev_raw = np.zeros((H, W), dtype=np.float32)
    with rasterio.open(os.path.join(city_dir, "elevation.tif")) as src:
        reproject(
            source=src.read(1).astype(np.float32),
            destination=elev_raw,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
        # Sample elevation at traversal points to get the normalization range,
        # matching exactly how the dataloader normalizes elevation.
        trav_reproj = traversal.to_crs(src.crs)
        trav_elev_pts = np.array(
            [v[0] for v in src.sample([(pt.x, pt.y) for pt in trav_reproj.geometry])]
        )

    elev_min, elev_max = float(trav_elev_pts.min()), float(trav_elev_pts.max())
    elev_norm = (elev_raw - elev_min) / (elev_max - elev_min)
    logger.info(f"Elevation range (traversal pts): [{elev_min:.1f}, {elev_max:.1f}] m")

    num_nlcd = len(_NLCD_CLASSES)  # 20
    num_gs = len(_GREEN_CLASSES)  # 8
    num_dims = num_nlcd + num_gs  # 28

    ###################################################
    # Steps 3–5: One-hot encoding, FFT convolution, temperature estimate
    # Processed one channel at a time to avoid storing 28 full-grid arrays.
    ###################################################

    # Pixel size in metres (y-direction, matching the dataloader convention)
    to_wgs84 = Transformer.from_crs(ref_crs, "EPSG:4326", always_xy=True)
    cx = ref_transform.c + ref_transform.a * (W / 2)
    cy = ref_transform.f + ref_transform.e * (H / 2)
    lon1, lat1 = to_wgs84.transform(cx, cy)
    lon2, lat2 = to_wgs84.transform(cx, cy + abs(ref_transform.e))
    _, _, pixel_size_m = Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
    logger.info(f"Pixel size: {pixel_size_m:.2f} m")

    # Zero-pad to reduce circular wrap-around artifacts at map edges
    pad = max(10, int(np.ceil(3.0 * lengthscale / pixel_size_m)))
    Hp, Wp = H + 2 * pad, W + 2 * pad

    # Build the isotropic exponential kernel in FFT wrap-around convention:
    # origin is at (0, 0) with negative-frequency indices wrapped to the end.
    ix = np.arange(Hp, dtype=float)
    iy = np.arange(Wp, dtype=float)
    ix = np.where(ix < Hp // 2, ix, ix - Hp) * pixel_size_m
    iy = np.where(iy < Wp // 2, iy, iy - Wp) * pixel_size_m
    r = np.sqrt(ix[:, None] ** 2 + iy[None, :] ** 2)
    kernel = np.exp(-r / lengthscale)
    kernel /= kernel.sum()
    kernel_fft = np.fft.rfft2(kernel)
    del ix, iy, r, kernel

    # Accumulate temperature channel-by-channel; write each greenspace class
    # cooling effect to its own file as soon as it is computed.
    # This avoids materialising the full (28, H, W) features/convolved arrays.
    temp_map = np.full((H, W), intercept, dtype=np.float32)
    temp_map += elevation_weight * elev_norm
    del elev_norm

    os.makedirs(output_dir, exist_ok=True)
    tif_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": W,
        "height": H,
        "count": 1,
        "crs": ref_crs,
        "transform": ref_transform,
        "compress": "lzw",
    }

    classes = list(_NLCD_CLASSES) + list(_GREEN_CLASSES)
    sources = [nlcd_data] * num_nlcd + [gs_data] * num_gs

    for c, (cls, src_data) in enumerate(zip(classes, sources)):
        # Build one-hot for this channel; negate greenspace (cooling effect)
        ch = (src_data == cls).astype(np.float32)
        if c >= num_nlcd:
            ch = -ch

        # Raw (point) contribution
        temp_map += beta[c] * ch
        if c >= num_nlcd:
            class_cooling = beta[c] * ch

        # Convolved contribution via FFT
        padded = np.pad(ch.astype(np.float64), pad)
        del ch
        conv_ch = np.fft.irfft2(np.fft.rfft2(padded) * kernel_fft, s=(Hp, Wp))[
            pad : pad + H, pad : pad + W
        ].astype(np.float32)
        del padded

        temp_map += beta[num_dims + c] * conv_ch
        if c >= num_nlcd:
            class_cooling += beta[num_dims + c] * conv_ch
            gs_out = os.path.join(output_dir, f"gs_cooling_{cls}_{args.time}.tif")
            with rasterio.open(gs_out, "w", **tif_profile) as dst:
                dst.write(class_cooling, 1)
                dst.update_tags(
                    1, description=f"Greenspace class {cls} cooling effect (Celsius)"
                )
            del class_cooling
        del conv_ch

        logger.info(f"Channel {c + 1}/{num_dims} done")

    ###################################################
    # Step 6: Save temperature; greenspace class files written during loop
    ###################################################

    out_path = os.path.join(output_dir, f"temperature_map_{args.time}.tif")
    with rasterio.open(out_path, "w", **tif_profile) as dst:
        dst.write(temp_map, 1)
        dst.update_tags(1, description="Estimated air temperature (Celsius)")

    logger.info(f"Saved temperature to {out_path}")
    logger.info(f"Saved {num_gs} greenspace class cooling maps to {output_dir}")


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
