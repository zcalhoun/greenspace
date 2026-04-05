"""
This script will process each of the cities and save the data in a format suitable for training.

"""

import os
import argparse
import json
from pathlib import Path

import numpy as np

import geopandas as gpd
import rasterio
from rasterio.windows import Window


def main(args):

    data_dir = Path(args.data_dir)

    if args.city is None:
        cities = os.listdir(data_dir)

        for city in cities:
            try:
                print(f"Processing city: {city}")
                process_city(data_dir, city, args.output_dir, args.window_size)
            except Exception as e:
                print(f"Error processing city {city}: {e}")
    else:
        process_city(data_dir, args.city, args.output_dir, args.window_size)


def process_city(data_dir, city, output_dir, window_size):
    """
    Iterates over am, af, pm and creates the necessary file structure for model training.
    """

    for time in ["am", "af", "pm"]:
        city_path = os.path.join(data_dir, city)

        temp_file = os.path.join(city_path, time, "trav.shp")
        nlcd_file = os.path.join(city_path, "nlcd.tif")
        greenspace_file = os.path.join(city_path, "greenspace.tif")
        ndvi_albedo_file = os.path.join(city_path, "ndvi_albedo.tif")
        elev_file = os.path.join(city_path, "elevation.tif")
        meta_file = os.path.join(city_path, time, "meta.json")
        meta = load_metadata(meta_file)

        for file in [
            temp_file,
            nlcd_file,
            greenspace_file,
            ndvi_albedo_file,
            elev_file,
        ]:
            if not os.path.exists(file):
                print(f"File {file} does not exist. Skipping.")
                return

        gdf = gpd.read_file(temp_file)

        # Initialize object with temperature data.
        temperature = extract_temperature(gdf, meta)
        obj = [{"temp": v.item()} for v in temperature]

        extract_coords(gdf, obj)
        extract_elevation(gdf, elev_file, obj)

        # Ensure all rasters are in the same CRS as the points
        with rasterio.open(nlcd_file) as src:
            if gdf.crs != src.crs:
                nlcd_gdf = gdf.to_crs(src.crs)

        with rasterio.open(ndvi_albedo_file) as src:
            if gdf.crs != src.crs:
                ndvi_albedo_gdf = gdf.to_crs(src.crs)

        with rasterio.open(greenspace_file) as src:
            if gdf.crs != src.crs:
                greenspace_gdf = gdf.to_crs(src.crs)

        parent_dir = os.path.join(output_dir, city, time)
        os.makedirs(parent_dir, exist_ok=True)
        for i, o in enumerate(obj):
            nlcd = extract_nlcd(nlcd_gdf.iloc[i], nlcd_file, window_size)
            greenspace = extract_greenspace(
                greenspace_gdf.iloc[i], greenspace_file, window_size
            )
            ndvi_albedo = extract_ndvi_albedo(
                ndvi_albedo_gdf.iloc[i], ndvi_albedo_file, window_size
            )

            if nlcd is None:
                continue
            if greenspace is None:
                continue
            if ndvi_albedo is None:
                continue

            o["nlcd_window"] = nlcd
            o["greenspace_window"] = greenspace
            o["ndvi_albedo_window"] = ndvi_albedo

            output_file = os.path.join(output_dir, city, time, f"{i}.json")

            with open(output_file, "w") as f:
                json.dump(o, f)


def extract_ndvi_albedo(point, ndvi_albedo_file, window_size):

    with rasterio.open(ndvi_albedo_file) as src:
        row, col = src.index(point.geometry.x, point.geometry.y)

        half_window = window_size // 2
        window = Window(col - half_window, row - half_window, window_size, window_size)

        try:
            # Read the window
            data = src.read((1, 2), window=window)
            if data.shape != (2, window_size, window_size):
                return None
            return data.tolist()
        except Exception as e:
            print(f"Warning: Could not extract value for point {idx}: {e}")
            return None


def extract_greenspace(point, greenspace_file, window_size):
    with rasterio.open(greenspace_file) as src:
        row, col = src.index(point.geometry.x, point.geometry.y)

        half_window = window_size // 2
        window = Window(col - half_window, row - half_window, window_size, window_size)

        try:
            data = src.read(1, window=window)
            if data.shape != (window_size, window_size):
                return None
            return data.tolist()
        except Exception as e:
            print(f"Warning: Could not extract greenspace value for point: {e}")
            return None


def extract_nlcd(point, nlcd_file, window_size):
    with rasterio.open(nlcd_file) as src:
        row, col = src.index(point.geometry.x, point.geometry.y)

        half_window = window_size // 2
        window = Window(col - half_window, row - half_window, window_size, window_size)

        try:
            data = src.read(1, window=window)
            if data.shape != (window_size, window_size):
                return None
            return data.tolist()
        except Exception as e:
            print(f"Warning: Could not extract NLCD value for point: {e}")
            return None


def extract_elevation(gdf, elev_file, obj):
    with rasterio.open(elev_file) as src:
        # Reproject points to raster CRS if needed
        if gdf.crs != src.crs:
            gdf_reproj = gdf.to_crs(src.crs)
        else:
            gdf_reproj = gdf

        coords = [
            (point.geometry.x, point.geometry.y) for _, point in gdf_reproj.iterrows()
        ]

        # Sample all points at once (more efficient)
        elevations = list(src.sample(coords, 1))

        # Min/max elevation for normalization
        elevations = np.stack(elevations).ravel()
        elev_min, elev_max = elevations.min(), elevations.max()
        elevations = (elevations - elev_min) / (elev_max - elev_min)

        for idx, elev in enumerate(elevations):
            obj[idx]["elev"] = elev.item()


def load_metadata(meta_path: str):
    """Load metadata from a JSON file."""
    with open(meta_path, "r") as f:
        return json.load(f)


def extract_temperature(gdf, meta):
    t = gdf[meta["temperature_variable"]].values

    if meta["unit"] == "fahrenheit":
        t = (t - 32) * 5.0 / 9.0

    return t


def extract_coords(gdf, obj):
    coords = [
        [g.geometry.x, g.geometry.y] for _, g in gdf.to_crs("EPSG:5070").iterrows()
    ]
    coords = np.array(coords)

    # Standardize the coordinates
    coords_mean = coords.mean(axis=0)
    coords_std = coords.std(axis=0)
    coords = (coords - coords_mean) / coords_std

    for i, o in enumerate(obj):
        o["coords"] = coords[i, :].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process the dataset and save it in a format suitable for training."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing the raw data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save the processed data.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=51,
        help="Size of the window to extract around each point.",
    )

    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="If specified, only implement for a specific city.",
    )
    arguments = parser.parse_args()

    main(arguments)
