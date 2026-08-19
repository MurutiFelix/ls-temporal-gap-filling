# src/preprocessing/raster_processor.py
"""
Geospatial Raster Processor for Landsat, AVHRR/MODIS, ERA5, and static modalities.
Handles raster loading, alignment to 30m Landsat grid, feature standardization,
and 3D-to-2D matrix flattening.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.preprocessing import StandardScaler


class RasterProcessor:
    """Handles raster loading, transformations, normalization, and feature array concatenation."""

    def __init__(self, config: dict):
        self.config = config
        self.num_bands = config["landsat"]["num_bands"]
        self.root_dir = Path(__file__).resolve().parents[2]  # repo root
        self.scaler = StandardScaler()
        self._scaler_fitted = False

    def normalize_reflectance(self, arr: np.ndarray) -> np.ndarray:
        """Scales optical reflectance strictly between 0.0 and 1.0."""
        return np.clip(arr, 0.0, 1.0)

    def _read_tif(self, path: Path) -> np.ndarray:
        """Reads a single-band GeoTIFF into a 2D array."""
        with rasterio.open(path) as src:
            return src.read(1)

    def _read_and_resample(self, path: Path, target_shape: tuple) -> np.ndarray:
        """Reads a GeoTIFF and resamples it (bilinear) to match target_shape (H, W)."""
        with rasterio.open(path) as src:
            data = src.read(
                1,
                out_shape=target_shape,
                resampling=Resampling.bilinear,
            )
        return data

    def find_landsat_file(self, band: str, year: int, month: int) -> Path:
        """Locates a Landsat band file: band_year_month_mission.tif — mission suffix varies."""
        mm = f"{month:02d}"
        pattern = f"{band.lower()}_{year}_{mm}_*.tif"
        matches = sorted(self.root_dir.glob(f"{self.config['paths']['landsat_dir']}/{pattern}"))
        if not matches:
            return None
        return matches[0]

    def find_era5_file(self, variable: str, year: int, month: int) -> Path:
        """Locates an ERA5 file: era5_{precip|temp}_year_month.tif"""
        mm = f"{month:02d}"
        key = f"era5_{variable}_dir"
        path = self.root_dir / self.config["paths"][key] / f"era5_{variable}_{year}_{mm}.tif"
        return path if path.exists() else None

    def find_avhrr_file(self, year: int, month: int) -> Path:
        """Locates an AVHRR file: avhrr_ndvi_year_month.tif (gap months only)."""
        mm = f"{month:02d}"
        path = self.root_dir / self.config["paths"]["avhrr_dir"] / f"avhrr_ndvi_{year}_{mm}.tif"
        return path if path.exists() else None

    def load_landsat_month(self, year: int, month: int) -> np.ndarray:
        """Loads all 7 Landsat bands for a given month, stacked as (H, W, 7). None if any missing."""
        bands = self.config["landsat"]["bands"]
        arrays = []

        for band in bands:
            path = self.find_landsat_file(band, year, month)
            if path is None:
                return None  # month has no Landsat coverage - a gap month
            arr = self._read_tif(path)
            arrays.append(arr)

        return np.stack(arrays, axis=-1)

    def load_era5_month(self, year: int, month: int, target_shape: tuple) -> np.ndarray:
        """Loads ERA5 precip + temp for a month, resampled to target_shape, stacked (H, W, 2)."""
        precip_path = self.find_era5_file("precip", year, month)
        temp_path = self.find_era5_file("temp", year, month)
        if precip_path is None or temp_path is None:
            return None

        precip = self._read_and_resample(precip_path, target_shape)
        temp = self._read_and_resample(temp_path, target_shape)
        return np.stack([precip, temp], axis=-1)

    def load_static(self, target_shape: tuple) -> np.ndarray:
        """Loads DEM (static, always available), resampled to target_shape."""
        dem_path = self.root_dir / self.config["paths"]["static_dir"] / "dem.tif"
        dem = self._read_and_resample(dem_path, target_shape)
        return dem[..., np.newaxis]  # (H, W, 1)

    def load_avhrr_month(self, year: int, month: int, target_shape: tuple) -> np.ndarray:
        """Loads AVHRR NDVI for evaluation-only comparison (gap months only). Resampled up to 30m."""
        path = self.find_avhrr_file(year, month)
        if path is None:
            return None
        return self._read_and_resample(path, target_shape)

    def get_pixel_coords(self, target_shape: tuple) -> np.ndarray:
        """
        Generates normalized (x, y) pixel-grid coordinate features, (H, W, 2), range 0-1.
        NOTE: this is relative pixel position within the AOI, not true geographic
        coordinates. Swap for src.utils.spatial.generate_spatial_coordinates(shape, bounds)
        if genuine UTM/lat-lon coordinates are preferred as model inputs.
        """
        h, w = target_shape
        xs = np.linspace(0.0, 1.0, w)
        ys = np.linspace(0.0, 1.0, h)
        grid_x, grid_y = np.meshgrid(xs, ys)
        return np.stack([grid_x, grid_y], axis=-1)

    def fit_scaler(self, X_raw: np.ndarray):
        """Fits StandardScaler on training X only — call once, before transform_features."""
        self.scaler.fit(X_raw)
        self._scaler_fitted = True

    def transform_features(self, X_raw: np.ndarray) -> np.ndarray:
        """Standardizes X using the scaler fitted on training data. Raises if not fitted."""
        if not self._scaler_fitted:
            raise RuntimeError("Call fit_scaler() on training data before transform_features().")
        return self.scaler.transform(X_raw)

    def assemble_feature_matrix(
        self,
        coarse_bands: np.ndarray,
        climate_data: np.ndarray,
        static_norms: np.ndarray,
    ) -> np.ndarray:
        """
        Legacy helper: flattens spatial dimensions and concatenates predictor arrays.
        Superseded in train.py/predict.py by explicit per-source concatenation
        (dem + coords + time + era5), kept here for backward compatibility.
        """
        n_samples = coarse_bands.shape[0]
        f_coarse = coarse_bands.reshape(n_samples, -1)
        f_climate = climate_data.reshape(n_samples, -1)
        f_norms = static_norms.reshape(n_samples, -1)
        return np.hstack([f_coarse, f_climate, f_norms])