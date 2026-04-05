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
            process_city(data_dir, city, args.output_dir, args.window_size)
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

        extract_nlcd(gdf, nlcd_file, obj, window_size)

        extract_greenspace(gdf, greenspace_file, obj, window_size)

        extract_ndvi_albedo(gdf, ndvi_albedo_file, obj, window_size)

        extract_elevation(gdf, elev_file, obj, window_size)

        filtered_obj = validate_obj(obj, window_size)

        output_path = os.path.join(output_dir, city, time)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i in range(len(filtered_obj)):
            output_file = os.path.join(output_path, f"{i}.json")
            with open(output_file, "w") as f:
                json.dump(filtered_obj[i], f)


def validate_obj(obj, window_size):
    filtered_obj = []
    for item in obj:
        if (
            "nlcd_window" in item
            and item["nlcd_window"] is not None
            and np.array(item["nlcd_window"]).shape != (window_size, window_size)
        ):
            continue
        if (
            "greenspace_window" in item
            and item["greenspace_window"] is not None
            and np.array(item["greenspace_window"]).shape != (window_size, window_size)
        ):
            continue
        if (
            "ndvi_albedo_window" in item
            and item["ndvi_albedo_window"] is not None
            and np.array(item["ndvi_albedo_window"]).shape
            != (2, window_size, window_size)
        ):
            continue
        filtered_obj.append(item)

    return filtered_obj


def extract_elevation(gdf, elev_file, obj, window_size):
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


def extract_ndvi_albedo(gdf, ndvi_albedo_file, obj, window_size):
    with rasterio.open(ndvi_albedo_file) as src:
        if gdf.crs != src.crs:
            gdf_reproj = gdf.to_crs(src.crs)
        else:
            gdf_reproj = gdf

        for idx, point in gdf_reproj.iterrows():
            # Get pixel coordinates
            row, col = src.index(point.geometry.x, point.geometry.y)

            # Calculate window around the point

            half_window = window_size // 2
            window = Window(
                col - half_window, row - half_window, window_size, window_size
            )

            try:
                # Read the window
                data = src.read((1, 2), window=window)

                obj[idx]["ndvi_albedo_window"] = data.tolist()
            except Exception as e:
                # Handle edge cases (points outside raster, etc.)
                obj[idx]["ndvi_albedo_window"] = None
                print(f"Warning: Could not extract value for point {idx}: {e}")


def extract_greenspace(gdf, greenspace_file, obj, window_size):
    # Greenspace file
    with rasterio.open(greenspace_file) as src:
        # Reproject points to raster CRS if needed
        if gdf.crs != src.crs:
            gdf_reproj = gdf.to_crs(src.crs)
        else:
            gdf_reproj = gdf

        for idx, point in gdf_reproj.iterrows():
            # Get pixel coordinates
            row, col = src.index(point.geometry.x, point.geometry.y)

            # Calculate window around the point
            half_window = window_size // 2
            window = Window(
                col - half_window, row - half_window, window_size, window_size
            )

            try:
                # Read the window
                data = src.read(1, window=window)

                # Handle nodata values
                # if src.nodata is not None:
                #     data = np.ma.masked_equal(data, src.nodata)

                obj[idx]["greenspace_window"] = data.tolist()
            except Exception as e:
                # Handle edge cases (points outside raster, etc.)
                obj[idx]["greenspace_window"] = None
                print(f"Warning: Could not extract value for point {idx}: {e}")


def extract_nlcd(gdf, nlcd_file, obj, window_size):
    """
    Iterate through the NLCD data and add the NLCD window.
    """
    with rasterio.open(nlcd_file) as src:
        # Reproject points to raster CRS if needed
        if gdf.crs != src.crs:
            gdf_reproj = gdf.to_crs(src.crs)
        else:
            gdf_reproj = gdf

        for idx, point in gdf_reproj.iterrows():
            # Get pixel coordinates
            row, col = src.index(point.geometry.x, point.geometry.y)

            # Calculate window around the point
            half_window = window_size // 2
            window = Window(
                col - half_window, row - half_window, window_size, window_size
            )

            try:
                # Read the window
                data = src.read(1, window=window)

                # Handle nodata values
                if src.nodata is not None:
                    data = np.ma.masked_equal(data, src.nodata)

                obj[idx]["nlcd_window"] = data.tolist()
            except Exception as e:
                # Handle edge cases (points outside raster, etc.)
                obj[idx]["nlcd_window"] = None
                print(f"Warning: Could not extract value for point {idx}: {e}")


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
