"""
This module takes a new approach to filtering out the buffer points, which
should how quickly the model trains.
"""

import os
import json

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.transform import AffineTransformer
from pyproj import Transformer, Geod

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _pixel_size_metres(raster_obj):
    """
    Return the pixel size in metres for a raster, regardless of its CRS.
    Steps one pixel in the y (latitude) direction from the raster midpoint,
    since GEE exports use the y-direction metres-per-degree (111,320 m/°) when
    converting a scale parameter to geographic pixel size. Measuring in x would
    give cos(lat)-compressed values at mid-latitudes.
    """
    t = raster_obj.transform
    bounds = raster_obj.bounds
    cx = (bounds.left + bounds.right) / 2
    cy = (bounds.bottom + bounds.top) / 2
    # Step one pixel upward (y direction)
    cy2 = cy + abs(t[4])

    to_wgs84 = Transformer.from_crs(raster_obj.crs, "EPSG:4326", always_xy=True)
    lon1, lat1 = to_wgs84.transform(cx, cy)
    lon2, lat2 = to_wgs84.transform(cx, cy2)

    geod = Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    return float(dist_m)


class GreenspaceDataset(Dataset):
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
    _GREEN_CLASSES = [2, 4, 6, 8, 10, 12, 13, 14]
    # Precomputed LUTs: raw raster value → class index (0-based).
    # Sized +2 so the last slot acts as a dump index (num_classes) for any
    # value clamped above the valid range or not present in the class list.
    _NLCD_LUT = torch.full(
        (max(_NLCD_CLASSES) + 2,), len(_NLCD_CLASSES), dtype=torch.long
    )
    for _idx, _val in enumerate(_NLCD_CLASSES):
        _NLCD_LUT[_val] = _idx

    _GS_LUT = torch.full(
        (max(_GREEN_CLASSES) + 2,), len(_GREEN_CLASSES), dtype=torch.long
    )
    for _idx, _val in enumerate(_GREEN_CLASSES):
        _GS_LUT[_val] = _idx

    def __init__(
        self,
        data_dir=None,
        greenspace=False,
        time="pm",
        window_size=500,  # Distance in meters.
        ndvi_albedo=True,
        non_negative=False,
    ):
        """
        Data_dir is the directory of the source data.
        greenspace indicates whether greenspace should ultimately be returned.
        """

        self.data_dir = data_dir
        self.greenspace = greenspace
        self.window_size = window_size
        self.return_greenspace = greenspace
        self.return_ndvi_albedo = ndvi_albedo
        self.non_negative = non_negative

        # First, we are going to load the coordinates
        temp_file = os.path.join(data_dir, time, "trav.shp")
        gdf = gpd.read_file(temp_file)

        # Second, we load the rasters.
        nlcd_ws = 2 * (window_size // 30) + 1
        nlcd = ClassificationRaster(os.path.join(data_dir, "nlcd.tif"), nlcd_ws)

        gs_ws = 2 * (window_size // 0.6) + 1
        greenspace = ClassificationRaster(
            os.path.join(data_dir, "greenspace.tif"), gs_ws
        )
        na_ws = 2 * (window_size // 10) + 1
        ndvi_albedo = NDVIAlbedo(os.path.join(data_dir, "ndvi_albedo.tif"), na_ws)

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

        if self.non_negative:
            self.init_temp = torch.tensor(float(self.temp.min()), dtype=torch.float32)
        else:
            self.init_temp = t
        self.window_size = [w.size(1) for w in s]
        self.num_dims = [w.size(0) for w in s]

    def __len__(self):
        return len(self.temp)

    @property
    def dimension_resolution(self):
        """
        Returns a 1D numpy array of metres-per-pixel for each feature channel,
        in the same order they are stacked in __getitem__:
          NLCD (20 channels) | [Greenspace (9 channels)] | NDVI + Albedo (2 channels)
        The pixel size is taken as the absolute x-pixel size from each raster's
        affine transform, which is in metres for projected CRS.
        """
        nlcd_res = _pixel_size_metres(self.nlcd)
        resolutions = [nlcd_res]
        if self.return_greenspace:
            gs_res = _pixel_size_metres(self.greenspace)
            resolutions.append(gs_res)
        if self.return_ndvi_albedo:
            ndvi_res = _pixel_size_metres(self.ndvi_albedo)
            resolutions.append(ndvi_res)
        return np.array(resolutions, dtype=np.float32)

    def _extract_coords(self, gdf):
        coords = [
            [g.geometry.x, g.geometry.y] for _, g in gdf.to_crs("EPSG:5070").iterrows()
        ]
        coords = np.array(coords)

        # Standardize the coordinates
        coords_mean = coords.mean(axis=0)
        # coords_std = coords.std(axis=0)
        coords = coords - coords_mean  # / coords_std

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

        nlcd_oh = self._create_one_hot(
            nlcd_window, self._NLCD_LUT, len(self._NLCD_CLASSES)
        ).permute(
            2, 0, 1
        )  # (C, H, W)

        if self.return_greenspace:
            greenspace_window = torch.from_numpy(self.greenspace[idx]).long()
            gs_oh = self._create_one_hot(
                greenspace_window, self._GS_LUT, len(self._GREEN_CLASSES)
            ).permute(
                2, 0, 1
            )  # (C, H, W)

            if self.non_negative:
                gs_oh = -gs_oh
            # Add the NDVI and albedo channels to the stacked window
            if self.return_ndvi_albedo:
                stacked_window = (nlcd_oh, gs_oh, ndvi_albedo_window)
            else:
                stacked_window = (nlcd_oh, gs_oh)
        else:
            stacked_window = (nlcd_oh, ndvi_albedo_window)

        return coords, elev, stacked_window, temp

    def _create_one_hot(self, window, lut, num_classes):
        # Clamp to LUT bounds; anything out-of-range or not in the class list
        # resolves to the dump index (num_classes) via the LUT initialisation.
        indices = lut[window.clamp(0, lut.size(0) - 1)]
        # One-hot has num_classes+1 channels; drop the last (dump) channel so
        # invalid pixels become all-zeros in the output.
        return F.one_hot(indices, num_classes=num_classes + 1).float()[
            ..., :num_classes
        ]


class CausalGreenspaceDataset(GreenspaceDataset):
    """
    This class just re-defines the reference classes
    """

    _NLCD_CLASSES = [
        11,
        12,
        21,
        22,
        23,
        24,
        31,
        81,
        82,
        90,
        95,
    ]
    _GREEN_CLASSES = [0, 2, 4, 6, 8, 10, 12, 13, 14]
    _NLCD_LUT = torch.full(
        (max(_NLCD_CLASSES) + 2,), len(_NLCD_CLASSES), dtype=torch.long
    )
    for _idx, _val in enumerate(_NLCD_CLASSES):
        _NLCD_LUT[_val] = _idx

    _GS_LUT = torch.full(
        (max(_GREEN_CLASSES) + 2,), len(_GREEN_CLASSES), dtype=torch.long
    )
    for _idx, _val in enumerate(_GREEN_CLASSES):
        _GS_LUT[_val] = _idx

    def __init__(
        self,
        data_dir=None,
        greenspace=True,
        time="pm",
        window_size=500,  # Distance in meters.
        ndvi_albedo=False,
        non_negative=False,
    ):
        super().__init__(
            data_dir,
            greenspace=greenspace,
            time=time,
            window_size=window_size,
            ndvi_albedo=ndvi_albedo,
        )


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
        width_buffer = np.abs(self.transform[0] * self.window_size)
        height_buffer = np.abs(self.transform[4] * self.window_size)

        gdf_reproj = gdf_reproj.cx[
            self.bounds.left + width_buffer : self.bounds.right - width_buffer,
            self.bounds.bottom + height_buffer : self.bounds.top - height_buffer,
        ]
        return gdf_reproj.to_crs(gdf.crs)

    def initialize_coords(self, gdf):
        gdf_reproj = gdf.to_crs(self.crs)
        transformer = rasterio.transform.AffineTransformer(self.transform)
        self.coords = np.column_stack(
            transformer.rowcol(gdf_reproj.geometry.x, gdf_reproj.geometry.y)
        )

    def __getitem__(self, idx):
        row, col = self.coords[idx]

        half_window = int(self.window_size // 2)
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
