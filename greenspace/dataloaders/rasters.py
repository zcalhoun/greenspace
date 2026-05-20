import numpy as np
import rasterio
import rasterio.transform


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
