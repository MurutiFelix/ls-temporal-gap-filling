# src/preprocessing/raster_processor.py
"""
Data Processing Engine for Temporal Gap-Filling.
Handles raster loading, bilinear resampling, standardization, and temporal neighbor retrieval.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.preprocessing import StandardScaler


class RasterProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.num_bands = config["landsat"]["num_bands"]
        self.root_dir = Path(__file__).resolve().parents[2]
        self.scaler = StandardScaler()
        self._scaler_fitted = False

    def normalize_reflectance(self, arr: np.ndarray) -> np.ndarray:
        """
        Scales target Landsat bands:
        Optical (Bands 0-5) -> Clipped [0.0, 1.0]
        Thermal (Band 6)   -> Scaled relative to 350K max
        """
        arr_norm = arr.copy()
        arr_norm[:, :6] = np.clip(arr_norm[:, :6], 0.0, 1.0)
        arr_norm[:, 6] = arr_norm[:, 6] / 350.0
        return arr_norm

    def _read_tif(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read(1)

    def _read_and_resample(self, path: Path, target_shape: tuple) -> np.ndarray:
        with rasterio.open(path) as src:
            data = src.read(
                1,
                out_shape=target_shape,
                resampling=Resampling.bilinear,
            )
        return data

    def find_landsat_file(self, band: str, year: int, month: int) -> Path:
        mm = f"{month:02d}"
        pattern = f"{band.lower()}_{year}_{mm}_*.tif"
        matches = sorted(
            self.root_dir.glob(
                f"{self.config['paths']['landsat_dir']}/{pattern}"
            )
        )
        return matches[0] if matches else None

    def find_era5_file(self, variable: str, year: int, month: int) -> Path:
        mm = f"{month:02d}"
        key = f"era5_{variable}_dir"
        path = (
            self.root_dir
            / self.config["paths"][key]
            / f"era5_{variable}_{year}_{mm}.tif"
        )
        return path if path.exists() else None

    def find_avhrr_file(self, year: int, month: int) -> Path:
        mm = f"{month:02d}"
        path = (
            self.root_dir
            / self.config["paths"]["avhrr_dir"]
            / f"avhrr_ndvi_{year}_{mm}.tif"
        )
        return path if path.exists() else None

    def load_landsat_month(self, year: int, month: int) -> np.ndarray:
        bands = self.config["landsat"]["bands"]
        arrays = []
        for band in bands:
            path = self.find_landsat_file(band, year, month)
            if path is None:
                return None
            arrays.append(self._read_tif(path))
        return np.stack(arrays, axis=-1)

    def load_temporal_neighbors(
        self, year: int, month: int, max_search_months: int = 12
    ) -> Dict[str, Any]:
        """
        Searches backward and forward to locate the nearest valid Landsat scenes
        and returns their data cubes alongside their temporal step deltas (dt).
        """
        def month_offset(y: int, m: int, offset: int) -> Tuple[int, int]:
            total_m = (y * 12 + (m - 1)) + offset
            res_y = total_m // 12
            res_m = (total_m % 12) + 1
            return res_y, res_m

        prev_cube, dt_prev = None, None
        for step in range(1, max_search_months + 1):
            py, pm = month_offset(year, month, -step)
            cube = self.load_landsat_month(py, pm)
            if cube is not None:
                prev_cube = cube
                dt_prev = step
                break

        next_cube, dt_next = None, None
        for step in range(1, max_search_months + 1):
            ny, nm = month_offset(year, month, step)
            cube = self.load_landsat_month(ny, nm)
            if cube is not None:
                next_cube = cube
                dt_next = step
                break

        return {
            "prev_cube": prev_cube,
            "dt_prev": dt_prev if dt_prev is not None else -1,
            "next_cube": next_cube,
            "dt_next": dt_next if dt_next is not None else -1,
        }

    def load_era5_month(
        self, year: int, month: int, target_shape: tuple
    ) -> np.ndarray:
        precip_path = self.find_era5_file("precip", year, month)
        temp_path = self.find_era5_file("temp", year, month)
        if precip_path is None or temp_path is None:
            return None
        precip = self._read_and_resample(precip_path, target_shape)
        temp = self._read_and_resample(temp_path, target_shape)
        return np.stack([precip, temp], axis=-1)

    def load_static(self, target_shape: tuple) -> np.ndarray:
        dem_path = (
            self.root_dir / self.config["paths"]["static_dir"] / "dem.tif"
        )
        dem = self._read_and_resample(dem_path, target_shape)
        return dem[..., np.newaxis]

    def load_avhrr_month(
        self, year: int, month: int, target_shape: tuple
    ) -> np.ndarray:
        path = self.find_avhrr_file(year, month)
        return self._read_and_resample(path, target_shape) if path else None

    def fit_scaler(self, X_train: np.ndarray):
        self.scaler.fit(X_train)
        self._scaler_fitted = True

    def transform_features(self, X_raw: np.ndarray) -> np.ndarray:
        if not self._scaler_fitted:
            raise RuntimeError(
                "Call fit_scaler() on training data before transform_features()."
            )
        return self.scaler.transform(X_raw)