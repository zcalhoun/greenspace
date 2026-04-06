"""
This module takes a new approach to filtering out the buffer points, which
should how quickly the model trains.
"""

import os
import json

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import AffineTransformer

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_NLCD_CLASSES = [
    11,
    12,
    21,
    22,
    23,
    24,
    31,
    41,
    42,
    43,
    51,
    52,
    71,
    72,
    73,
    74,
    81,
    82,
    90,
    95,
]
_GREEN_CLASSES = [0, 2, 4, 6, 8, 10, 12, 13, 14]


class GreenspaceDataset(Dataset):
    """ """

    def __init__(self, data_dir=None, greenspace=False, time="pm", window_size=51):
        """
        Data_dir is the directory of the source data.
        greenspace indicates whether greenspace should ultimately be returned.
        """

        self.data_dir = data_dir
        self.greenspace = greenspace
        self.window_size = window_size
        self.return_greenspace = greenspace

        # First, we are going to load the coordinates
        temp_file = os.path.join(data_dir, time, "trav.shp")
        gdf = gpd.read_file(temp_file)

        # Second, we load the rasters.
        nlcd = ClassificationRaster(os.path.join(data_dir, "nlcd.tif"), window_size)
        greenspace = ClassificationRaster(
            os.path.join(data_dir, "greenspace.tif"), window_size
        )
        ndvi_albedo = NDVIAlbedo(os.path.join(data_dir, "ndvi_albedo.tif"), window_size)

        # For each of the rasters, we filter out the coordinates that do not
        # have a complete dataset.
        for raster in [nlcd, greenspace, ndvi_albedo]:
            gdf = raster.filter_gdf(gdf)

        for raster in [nlcd, greenspace, ndvi_albedo]:
            raster.initialize_coords(gdf)

        self.nlcd = nlcd
        self.greenspace = greenspace
        self.ndvi_albedo = ndvi_albedo
        self.elevation = self._extract_elevation(gdf)
        self.temp = self._extract_temperature(gdf, time)
        self.coords = self._extract_coords(gdf)

        _, _, s, t = self.__getitem__(0)

        self.init_temp = t
        self.window_size = s.size(1)
        self.num_dims = s.size(0)

    def __len__(self):
        return len(self.temp)

    def _extract_coords(self, gdf):
        coords = [
            [g.geometry.x, g.geometry.y] for _, g in gdf.to_crs("EPSG:5070").iterrows()
        ]
        coords = np.array(coords)

        # Standardize the coordinates
        coords_mean = coords.mean(axis=0)
        coords_std = coords.std(axis=0)
        coords = (coords - coords_mean) / coords_std

        return coords

    def _extract_temperature(self, gdf, time):
        temp_file = os.path.join(self.data_dir, time, "trav.shp")
        meta_file = os.path.join(self.data_dir, time, "meta.json")
        meta = load_metadata(meta_file)

        t = gdf[meta["temperature_variable"]].values

        if meta["unit"] == "fahrenheit":
            t = (t - 32) * 5.0 / 9.0

        return t

    def _extract_elevation(self, gdf):
        elev_file = os.path.join(self.data_dir, "elevation.tif")
        with rasterio.open(elev_file) as src:
            # Reproject points to raster CRS if needed
            if gdf.crs != src.crs:
                gdf_reproj = gdf.to_crs(src.crs)
            else:
                gdf_reproj = gdf

            coords = [
                (point.geometry.x, point.geometry.y)
                for _, point in gdf_reproj.iterrows()
            ]

            # Sample all points at once (more efficient)
            elevations = list(src.sample(coords, 1))

        # Normalize elevation
        elevations = np.stack(elevations).ravel()
        elev_min, elev_max = elevations.min(), elevations.max()
        elevations = (elevations - elev_min) / (elev_max - elev_min)

        return elevations

    def __getitem__(self, idx):

        coords = torch.tensor(self.coords[idx], dtype=torch.float32)
        elev = torch.tensor(self.elevation[idx], dtype=torch.float32).unsqueeze(
            0
        )  # Add channel dimension
        nlcd_window = torch.from_numpy(self.nlcd[idx]).long()
        ndvi_albedo_window = torch.from_numpy(self.ndvi_albedo[idx]).float()
        # Clip ndvi/albedo values to [0, 1]
        ndvi_albedo_window = torch.clamp(ndvi_albedo_window, 0.0, 1.0)
        temp = torch.tensor(self.temp[idx], dtype=torch.float32)

        nlcd_oh = self._create_one_hot(nlcd_window, _NLCD_CLASSES).permute(
            2, 0, 1
        )  # (C, H, W)

        if self.return_greenspace:
            greenspace_window = torch.from_numpy(self.greenspace[idx]).long()
            gs_oh = self._create_one_hot(greenspace_window, _GREEN_CLASSES).permute(
                2, 0, 1
            )  # (C, H, W)

            # Add the NDVI and albedo channels to the stacked window
            stacked_window = torch.cat([nlcd_oh, gs_oh, ndvi_albedo_window], dim=0)
        else:
            stacked_window = torch.cat([nlcd_oh, ndvi_albedo_window], dim=0)

        return coords, elev, stacked_window, temp

    def _create_one_hot(self, window, class_values):
        indices = torch.zeros_like(window)
        for idx, val in enumerate(class_values):
            indices[window == val] = idx
        return F.one_hot(indices, num_classes=len(class_values)).float()


def load_metadata(meta_path: str):
    """Load metadata from a JSON file."""
    with open(meta_path, "r") as f:
        return json.load(f)


class RasterObject:
    """
    A wrapper around rasters to facilitate selecting the data.
    """

    def __init__(self, data_dir, window_size):
        self.data_dir = data_dir
        self.window_size = window_size
        self.data = None

        with rasterio.open(self.data_dir) as src:
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds

    def filter_gdf(self, gdf):
        """
        Filters out the points in the gdf that do not have a complete dataset
        in the raster.
        """
        # Transform the gdf to the same CRS as the raster
        gdf_reproj = gdf.to_crs(self.crs)
        width_buffer = self.transform[0] * self.window_size
        height_buffer = self.transform[4] * self.window_size

        gdf_reproj = gdf_reproj.cx[
            self.bounds.left + width_buffer : self.bounds.right - width_buffer,
            self.bounds.bottom + height_buffer : self.bounds.top - height_buffer,
        ]
        return gdf_reproj.to_crs(gdf.crs)

    def initialize_coords(self, gdf):
        transformer = rasterio.transform.AffineTransformer(self.transform)
        self.coords = np.column_stack(
            transformer.rowcol(gdf.geometry.x, gdf.geometry.y)
        )

    def __getitem__(self, idx):
        row, col = self.coords[idx]

        half_window = self.window_size // 2
        return self.data[
            row - half_window : row + half_window + 1,
            col - half_window : col + half_window + 1,
        ]


class ClassificationRaster(RasterObject):
    """
    A wrapper around the NLCD raster to facilitate selecting the data.
    """

    def __init__(self, data_dir, window_size):
        super().__init__(data_dir, window_size)
        with rasterio.open(self.data_dir) as src:
            self.data = src.read(1)


class NDVIAlbedo(RasterObject):
    """
    A wrapper around the NDVI/Albedo raster to facilitate selecting the data.
    """

    def __init__(self, data_dir, window_size):
        super().__init__(data_dir, window_size)
        with rasterio.open(self.data_dir) as src:
            self.data = src.read((1, 2))

    def __getitem__(self, idx):
        row, col = self.coords[idx]

        half_window = self.window_size // 2
        return self.data[
            :,
            row - half_window : row + half_window + 1,
            col - half_window : col + half_window + 1,
        ]
