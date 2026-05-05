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


def _next_fast_fft_size(n: int) -> int:
    """Return the smallest power-of-two >= n for efficient FFT."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_exp_kernel(half_k: int, pixel_size_m: float, lengthscale: float) -> torch.Tensor:
    """
    Build a normalised 2-D exponential decay kernel.

    Args:
        half_k:       half-width of the kernel in pixels
        pixel_size_m: metres per pixel (after any pooling)
        lengthscale:  decay lengthscale in metres

    Returns:
        (2*half_k+1, 2*half_k+1) float tensor that sums to 1.
    """
    ys = torch.arange(-half_k, half_k + 1, dtype=torch.float32)
    xs = torch.arange(-half_k, half_k + 1, dtype=torch.float32)
    ys, xs = torch.meshgrid(ys, xs, indexing="ij")
    dist_m = torch.sqrt(ys ** 2 + xs ** 2) * pixel_size_m
    kernel = torch.exp(-dist_m / lengthscale)
    kernel = kernel / kernel.sum()
    return kernel


def _fft_convolve(feature_raster: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Convolve a multi-channel raster with a 2-D kernel via FFT.

    The kernel is assumed to have odd spatial dimensions with its centre at
    the middle pixel. Zero-padding is applied so the result is free of
    circular-convolution artefacts. The output has the same spatial size as
    the input.

    Args:
        feature_raster: (C, H, W) float tensor
        kernel:         (kH, kW) float tensor

    Returns:
        (C, H, W) float tensor
    """
    C, H, W = feature_raster.shape
    kH, kW = kernel.shape
    half_kH = kH // 2
    half_kW = kW // 2

    pad_H = _next_fast_fft_size(H + kH - 1)
    pad_W = _next_fast_fft_size(W + kW - 1)

    # Place kernel at top-left then roll so its centre lands at (0, 0) —
    # this converts correlation to the circular convolution that FFT computes.
    kernel_padded = torch.zeros(pad_H, pad_W)
    kernel_padded[:kH, :kW] = kernel
    kernel_padded = torch.roll(kernel_padded, shifts=(-half_kH, -half_kW), dims=(0, 1))
    K_fft = torch.fft.rfft2(kernel_padded)           # (pad_H, pad_W//2+1)

    # Batch-FFT all channels simultaneously.
    raster_padded = torch.zeros(C, pad_H, pad_W)
    raster_padded[:, :H, :W] = feature_raster
    X_fft = torch.fft.rfft2(raster_padded)           # (C, pad_H, pad_W//2+1)

    conv = torch.fft.irfft2(
        X_fft * K_fft.unsqueeze(0), s=(pad_H, pad_W)
    )                                                 # (C, pad_H, pad_W)

    # Crop back to original spatial size.
    #
    # Rolling the kernel by (-half_kH, -half_kW) shifts the entire convolution
    # output by the same amount:
    #
    #   rolled_conv[r] == unrolled_conv[r + half_k]
    #
    # The weighted sum for raster position i is unrolled_conv[i + half_k]
    # = rolled_conv[i].  So the correct starting index after rolling is 0,
    # not half_k.  Cropping at [half_k : half_k + H] would apply a second
    # half_k shift, placing each feature ~window_size metres from its true
    # spatial location.
    return conv[:, :H, :W].contiguous()


def _one_hot_pool_raster(
    data: np.ndarray,
    lut: torch.Tensor,
    num_classes: int,
    pool_factor: int,
    strip_rows: int = 10,
) -> torch.Tensor:
    """
    One-hot encode a 2-D integer raster and spatially average-pool it,
    processing in horizontal strips to avoid materialising the full one-hot
    tensor at once.

    For a 30 k × 30 k greenspace raster with 9 classes, the full float32
    one-hot tensor would be ~40 GB; strip processing caps peak usage at
    roughly (strip_rows * pool_factor * W * num_classes * 4) bytes.

    Args:
        data:        (H, W) numpy int array of raw raster values
        lut:         1-D LUT mapping raw values → class indices (0-based);
                     values not in the LUT map to num_classes (dump channel).
        num_classes: number of valid classes (dump channel excluded from output)
        pool_factor: spatial downsampling factor; 1 = no pooling
        strip_rows:  number of *output* rows to process per strip

    Returns:
        (num_classes, H // pool_factor, W // pool_factor) float tensor
    """
    H, W = data.shape
    H_trim = (H // pool_factor) * pool_factor
    W_trim = (W // pool_factor) * pool_factor

    out_H = H_trim // pool_factor
    out_W = W_trim // pool_factor
    result = torch.zeros(num_classes, out_H, out_W)

    in_strip = strip_rows * pool_factor
    for r_in in range(0, H_trim, in_strip):
        r_in_end = min(r_in + in_strip, H_trim)
        strip = torch.from_numpy(data[r_in:r_in_end, :W_trim].copy()).long()
        indices = lut[strip.clamp(0, lut.size(0) - 1)]         # (strip_H, W_trim)
        oh = F.one_hot(indices, num_classes=num_classes + 1).float()
        oh = oh[..., :num_classes].permute(2, 0, 1)            # (C, strip_H, W_trim)
        if pool_factor > 1:
            oh = F.avg_pool2d(
                oh.unsqueeze(0), kernel_size=pool_factor, stride=pool_factor
            ).squeeze(0)
        r_out = r_in // pool_factor
        result[:, r_out : r_out + oh.shape[1], :out_W] = oh

    return result


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
        gs_downsample=1,
    ):
        """
        Data_dir is the directory of the source data.
        greenspace indicates whether greenspace should ultimately be returned.
        """

        self.data_dir = data_dir
        self.greenspace = greenspace
        self.window_size_m = window_size      # spatial extent in metres
        self.return_greenspace = greenspace
        self.return_ndvi_albedo = ndvi_albedo
        self.non_negative = non_negative
        self.gs_downsample = int(gs_downsample)

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

        # Compute num_dims explicitly — avoids a costly __getitem__ call at init.
        self.num_dims = [len(self._NLCD_CLASSES)]
        if self.return_greenspace:
            self.num_dims.append(len(self._GREEN_CLASSES))
        if self.return_ndvi_albedo:
            self.num_dims.append(2)

        if self.non_negative:
            self.init_temp = torch.tensor(float(self.temp.min()), dtype=torch.float32)
        else:
            self.init_temp = torch.tensor(float(self.temp[0]), dtype=torch.float32)

        self._precompute_point_features()

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

        # Standardize the coordinates.  Store the mean so that get_raster_tile()
        # can apply the same normalisation to full-raster pixel centres.
        self.coords_mean = coords.mean(axis=0)
        coords = (coords - self.coords_mean) / 1000  # put it in KM

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
        self.elev_path = elev_file   # stored for full-raster prediction
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

        # Normalize elevation; store stats so get_raster_tile() uses the same scale.
        elevations = np.stack(elevations).ravel().astype(np.float32)
        # Replace nodata: rasterio returns the raster's nodata value for out-of-bounds
        # pixels; it may be a large sentinel (e.g. -9999) or NaN.  Replace both with
        # NaN so nanmin/nanmax ignore them, then fill NaN points with the mean.
        with rasterio.open(elev_file) as _src:
            _nodata = _src.nodata
        if _nodata is not None:
            elevations[elevations == _nodata] = np.nan
        valid_mask = np.isfinite(elevations)
        if valid_mask.any():
            self.elev_min = float(np.nanmin(elevations))
            self.elev_max = float(np.nanmax(elevations))
            fill = float(np.nanmean(elevations))
        else:
            self.elev_min, self.elev_max, fill = 0.0, 1.0, 0.0
        elevations = np.where(valid_mask, elevations, fill)
        elevations = (elevations - self.elev_min) / max(self.elev_max - self.elev_min, 1e-8)

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

            if self.gs_downsample > 1:
                # Reduce the greenspace window resolution before storing.
                # avg_pool2d expects (N, C, H, W); add/remove the batch dim.
                gs_oh = F.avg_pool2d(
                    gs_oh.unsqueeze(0),
                    kernel_size=self.gs_downsample,
                    stride=self.gs_downsample,
                ).squeeze(0)

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

    # ------------------------------------------------------------------
    # Full-raster FFT feature precomputation
    # ------------------------------------------------------------------

    def _precompute_point_features(self):
        """
        Precompute the per-point (centre-pixel) features for every sample.
        These do not depend on the kernel lengthscale, so they are computed
        once at dataset initialisation.

        Stores ``self._point_features`` of shape (N, sum(num_dims)).
        """
        parts = []

        # NLCD — look up class index at each training point, then one-hot.
        rows = self.nlcd.coords[:, 0]
        cols = self.nlcd.coords[:, 1]
        raw = torch.from_numpy(self.nlcd.data[rows, cols].copy()).long()
        idx = self._NLCD_LUT[raw.clamp(0, self._NLCD_LUT.size(0) - 1)]
        nlcd_oh = F.one_hot(idx, num_classes=len(self._NLCD_CLASSES) + 1).float()
        parts.append(nlcd_oh[:, : len(self._NLCD_CLASSES)])   # (N, 20)

        if self.return_greenspace:
            rows = self.greenspace.coords[:, 0]
            cols = self.greenspace.coords[:, 1]
            raw = torch.from_numpy(self.greenspace.data[rows, cols].copy()).long()
            idx = self._GS_LUT[raw.clamp(0, self._GS_LUT.size(0) - 1)]
            gs_oh = F.one_hot(idx, num_classes=len(self._GREEN_CLASSES) + 1).float()
            gs_oh = gs_oh[:, : len(self._GREEN_CLASSES)]       # (N, 9)
            if self.non_negative:
                gs_oh = -gs_oh
            parts.append(gs_oh)

        if self.return_ndvi_albedo:
            rows = self.ndvi_albedo.coords[:, 0]
            cols = self.ndvi_albedo.coords[:, 1]
            # ndvi_albedo.data is (2, H, W)
            na = torch.from_numpy(
                self.ndvi_albedo.data[:, rows, cols].T.copy()
            ).float()                                           # (N, 2)
            na = torch.nan_to_num(na, nan=0.0)
            na = torch.clamp(na, 0.0, 1.0)
            parts.append(na)

        self._point_features = torch.cat(parts, dim=1)         # (N, sum(num_dims))

    def precompute_features(self, lengthscale: float):
        """
        FFT-convolve every raster layer with the exponential decay kernel at
        the given lengthscale and store the result for fast point lookup.

        The precomputed tensors are cached: repeated calls with the same
        lengthscale are free.

        Args:
            lengthscale: kernel decay lengthscale in metres.
        """
        if getattr(self, "_precomputed_lengthscale", None) == lengthscale:
            return

        print(f"[GreenspaceDataset] Precomputing feature rasters "
              f"(lengthscale={lengthscale:.1f} m) …", flush=True)

        self._weighted_rasters = []   # list of (raster_obj, tensor (C,H,W), pool_factor)

        # --- NLCD (30 m native, no pooling needed) ---
        nlcd_px = _pixel_size_metres(self.nlcd)
        half_k = max(1, int(self.window_size_m / nlcd_px))
        kernel = _build_exp_kernel(half_k, nlcd_px, lengthscale)
        nlcd_oh = _one_hot_pool_raster(
            self.nlcd.data, self._NLCD_LUT, len(self._NLCD_CLASSES),
            pool_factor=1,
        )                                                        # (20, H, W)
        nlcd_weighted = _fft_convolve(nlcd_oh, kernel)          # (20, H, W)
        self._weighted_rasters.append((self.nlcd, nlcd_weighted, 1))

        # --- Greenspace (0.6 m native, pool by gs_downsample) ---
        if self.return_greenspace:
            gs_native_px = _pixel_size_metres(self.greenspace)
            gs_pooled_px = gs_native_px * self.gs_downsample
            half_k = max(1, int(self.window_size_m / gs_pooled_px))
            kernel = _build_exp_kernel(half_k, gs_pooled_px, lengthscale)
            gs_oh = _one_hot_pool_raster(
                self.greenspace.data, self._GS_LUT, len(self._GREEN_CLASSES),
                pool_factor=self.gs_downsample,
            )                                                    # (9, H', W')
            if self.non_negative:
                gs_oh = -gs_oh
            gs_weighted = _fft_convolve(gs_oh, kernel)          # (9, H', W')
            self._weighted_rasters.append(
                (self.greenspace, gs_weighted, self.gs_downsample)
            )

        # --- NDVI / Albedo (10 m native, no pooling needed) ---
        if self.return_ndvi_albedo:
            na_px = _pixel_size_metres(self.ndvi_albedo)
            half_k = max(1, int(self.window_size_m / na_px))
            kernel = _build_exp_kernel(half_k, na_px, lengthscale)
            na_data = torch.from_numpy(
                self.ndvi_albedo.data.astype(np.float32).copy()
            )                                                    # (2, H, W)
            # Replace nodata (NaN or out-of-range) with 0 before FFT.
            # A single NaN pixel would poison the entire frequency domain.
            na_data = torch.nan_to_num(na_data, nan=0.0)
            na_data = torch.clamp(na_data, 0.0, 1.0)
            na_weighted = _fft_convolve(na_data, kernel)        # (2, H, W)
            self._weighted_rasters.append((self.ndvi_albedo, na_weighted, 1))

        self._precomputed_lengthscale = lengthscale
        print("[GreenspaceDataset] Done.", flush=True)

    def get_all_features(self, indices: np.ndarray):
        """
        Vectorised feature lookup for an array of sample indices.

        Must call :meth:`precompute_features` before this method.

        Returns:
            coords: (N, 2) float tensor of standardised coordinates
            X:      (N, sum(num_dims)*2 + 1) float tensor
            y:      (N,) float tensor of temperatures
        """
        assert hasattr(self, "_weighted_rasters"), (
            "Call precompute_features(lengthscale) before get_all_features()."
        )

        weighted_parts = []
        for raster_obj, feat_raster, pool_factor in self._weighted_rasters:
            rows = raster_obj.coords[indices, 0] // pool_factor
            cols = raster_obj.coords[indices, 1] // pool_factor
            H, W = feat_raster.shape[1], feat_raster.shape[2]
            rows = np.clip(rows, 0, H - 1)
            cols = np.clip(cols, 0, W - 1)
            weighted_parts.append(feat_raster[:, rows, cols].T)  # (N, C)

        point_feats = self._point_features[indices]              # (N, sum(num_dims))
        elev = torch.tensor(
            self.elevation[indices], dtype=torch.float32
        ).unsqueeze(1)                                           # (N, 1)

        X = torch.cat([*weighted_parts, point_feats, elev], dim=1)
        coords = torch.tensor(self.coords[indices], dtype=torch.float32)
        y = torch.tensor(self.temp[indices], dtype=torch.float32)
        return coords, X, y

    def get_raster_tile(self, row_start: int, row_end: int):
        """
        Build feature vectors for every pixel in NLCD rows [row_start, row_end).

        Must call :meth:`precompute_features` before this method.

        The feature layout matches :meth:`get_all_features` exactly:
        ``[weighted_feats | point_feats | elev]`` so the same normalisation
        and ``model.beta`` apply unchanged.

        Args:
            row_start: first NLCD row (inclusive)
            row_end:   last NLCD row (exclusive)

        Returns:
            coords: (N, 2) float32 standardised EPSG:5070 coordinates (km),
                    where N = (row_end - row_start) * W
            X:      (N, D) float32 feature tensor
        """
        assert hasattr(self, "_weighted_rasters"), (
            "Call precompute_features(lengthscale) before get_raster_tile()."
        )

        H, W = self.nlcd.data.shape
        rows_arr = np.repeat(np.arange(row_start, row_end), W)  # (N,)
        cols_arr = np.tile(np.arange(W), row_end - row_start)   # (N,)

        # Convert NLCD pixel centres to NLCD CRS map coordinates once.
        nlcd_aff = AffineTransformer(self.nlcd.transform)
        x_nlcd, y_nlcd = nlcd_aff.xy(rows_arr.tolist(), cols_arr.tolist())
        x_nlcd = np.asarray(x_nlcd)
        y_nlcd = np.asarray(y_nlcd)

        def _to_raster_rowcol(raster_obj):
            """Convert NLCD CRS coords to pixel (row, col) in raster_obj's space."""
            if raster_obj.crs != self.nlcd.crs:
                proj = Transformer.from_crs(
                    self.nlcd.crs, raster_obj.crs, always_xy=True
                )
                xr, yr = proj.transform(x_nlcd, y_nlcd)
            else:
                xr, yr = x_nlcd, y_nlcd
            aff = AffineTransformer(raster_obj.transform)
            r, c = aff.rowcol(xr.tolist(), yr.tolist())
            return np.asarray(r), np.asarray(c)

        # --- Weighted features (from precomputed FFT rasters) ---
        weighted_parts = []
        for raster_obj, feat_raster, pool_factor in self._weighted_rasters:
            fH, fW = feat_raster.shape[1], feat_raster.shape[2]
            if raster_obj is self.nlcd:
                # Same pixel grid — no reprojection needed.
                r = np.clip(rows_arr, 0, fH - 1)
                c = np.clip(cols_arr, 0, fW - 1)
            else:
                r_raw, c_raw = _to_raster_rowcol(raster_obj)
                r = np.clip(r_raw // pool_factor, 0, fH - 1)
                c = np.clip(c_raw // pool_factor, 0, fW - 1)
            weighted_parts.append(feat_raster[:, r, c].T)   # (N, C)

        # --- Point features (one-hot at each pixel centre) ---
        point_parts = []

        raw_nlcd = torch.from_numpy(self.nlcd.data[rows_arr, cols_arr].copy()).long()
        idx_nlcd = self._NLCD_LUT[raw_nlcd.clamp(0, self._NLCD_LUT.size(0) - 1)]
        nlcd_oh = F.one_hot(idx_nlcd, num_classes=len(self._NLCD_CLASSES) + 1).float()
        point_parts.append(nlcd_oh[:, : len(self._NLCD_CLASSES)])

        if self.return_greenspace:
            r_gs, c_gs = _to_raster_rowcol(self.greenspace)
            r_gs = np.clip(r_gs, 0, self.greenspace.data.shape[0] - 1)
            c_gs = np.clip(c_gs, 0, self.greenspace.data.shape[1] - 1)
            raw_gs = torch.from_numpy(
                self.greenspace.data[r_gs, c_gs].copy()
            ).long()
            idx_gs = self._GS_LUT[raw_gs.clamp(0, self._GS_LUT.size(0) - 1)]
            gs_oh = F.one_hot(
                idx_gs, num_classes=len(self._GREEN_CLASSES) + 1
            ).float()
            gs_oh = gs_oh[:, : len(self._GREEN_CLASSES)]
            if self.non_negative:
                gs_oh = -gs_oh
            point_parts.append(gs_oh)

        if self.return_ndvi_albedo:
            r_na, c_na = _to_raster_rowcol(self.ndvi_albedo)
            r_na = np.clip(r_na, 0, self.ndvi_albedo.data.shape[1] - 1)
            c_na = np.clip(c_na, 0, self.ndvi_albedo.data.shape[2] - 1)
            na = torch.from_numpy(
                self.ndvi_albedo.data[:, r_na, c_na].T.copy()
            ).float()
            na = torch.nan_to_num(na, nan=0.0)
            na = torch.clamp(na, 0.0, 1.0)
            point_parts.append(na)

        # --- Elevation (sample from elevation raster at NLCD pixel centres) ---
        with rasterio.open(self.elev_path) as elev_src:
            if elev_src.crs != self.nlcd.crs:
                to_elev = Transformer.from_crs(
                    self.nlcd.crs, elev_src.crs, always_xy=True
                )
                x_e, y_e = to_elev.transform(x_nlcd, y_nlcd)
            else:
                x_e, y_e = x_nlcd, y_nlcd
            elev_vals = np.array(
                list(elev_src.sample(zip(x_e, y_e), 1))
            ).ravel()
        elev_norm = (elev_vals - self.elev_min) / max(
            self.elev_max - self.elev_min, 1e-8
        )
        elev_t = torch.tensor(elev_norm, dtype=torch.float32).unsqueeze(1)  # (N,1)

        # --- Coordinates: NLCD CRS → EPSG:5070 (if needed) → standardise ---
        # coords_mean is in EPSG:5070 (computed from gdf.to_crs("EPSG:5070")).
        if self.nlcd.crs.to_epsg() == 5070:
            x_5070, y_5070 = x_nlcd, y_nlcd
        else:
            to_5070 = Transformer.from_crs(
                self.nlcd.crs, "EPSG:5070", always_xy=True
            )
            x_5070, y_5070 = to_5070.transform(x_nlcd, y_nlcd)
        coords_km = (
            np.column_stack([x_5070, y_5070]) - self.coords_mean
        ) / 1000.0
        coords_t = torch.tensor(coords_km, dtype=torch.float32)

        X = torch.cat([*weighted_parts, *point_parts, elev_t], dim=1)
        return coords_t, X


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
