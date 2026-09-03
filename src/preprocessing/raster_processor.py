from __future__ import annotations

# src/preprocessing/raster_processor.py
"""
Central raster discovery, loading, and spatial alignment utilities.

Handles the 7-band Landsat reordering from source to model order:
Source Order: 1. Blue, 2. Green, 3. Red, 4. NIR, 5. SWIR1, 6. SWIR2, 7. Thermal
Model Order:  Red, Green, Blue, NIR, SWIR1, SWIR2, Thermal
"""


"""
Central raster discovery, loading, and spatial alignment utilities.
This module is responsible for:

Resolving project paths from config.yaml
Discovering monthly Landsat observations stored as seven-band GeoTIFFs
Validating complete Landsat observations and source band structure
Explicitly mapping Landsat source bands to the model band order
Detecting missing Landsat months
Defining and caching a common target grid
Reprojecting and resampling rasters to the target grid
Converting source NoData and invalid values to NaN
Loading complete Landsat monthly raster cubes
Loading ERA5 precipitation and temperature predictors
Loading and caching the static DEM
Retrieving the nearest previous and next Landsat observations
Computing temporal distances and Landsat availability indicators
Loading AVHRR and MODIS NDVI evaluation data

Landsat observations are stored as one stacked seven-band GeoTIFF per
month using the following source raster band order:
Source Band 1: Blue
Source Band 2: Green
Source Band 3: Red
Source Band 4: NIR
Source Band 5: SWIR1
Source Band 6: SWIR2
Source Band 7: Thermal

The model uses the following band order:
Model Band 0: Red
Model Band 1: Green
Model Band 2: Blue
Model Band 3: NIR
Model Band 4: SWIR1
Model Band 5: SWIR2
Model Band 6: Thermal

Expected Landsat filename pattern:
landsat_YYYY_MM_lX.tif
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.warp import reproject

EXPECTED_LANDSAT_BANDS = 7

MODEL_BAND_ORDER = (
    "Red",
    "Green",
    "Blue",
    "NIR",
    "SWIR1",
    "SWIR2",
    "Thermal",
)

# 1-based source GeoTIFF band indexing -> model band mapping
SOURCE_TO_MODEL_INDEX = {
    "Red": 3,
    "Green": 2,
    "Blue": 1,
    "NIR": 4,
    "SWIR1": 5,
    "SWIR2": 6,
    "Thermal": 7,
}


class RasterProcessor:
    """
    Handles raster discovery, spatial alignment, Landsat temporal
    neighbourhood loading, predictor loading, and evaluation-raster loading.
    """

    def __init__(
        self,
        config_or_path: Union[str, Path, Dict[str, Any]]
    ) -> None:

        if isinstance(config_or_path, (str, Path)):
            self.config_path = Path(config_or_path)

            if not self.config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_path}"
                )

            with open(
                self.config_path,
                "r",
                encoding="utf-8"
            ) as file:
                self.config = yaml.safe_load(file)

            self.project_root = self.config_path.parent.parent

        elif isinstance(config_or_path, dict):
            self.config = config_or_path
            self.config_path = Path("config.yaml")
            self.project_root = Path.cwd()

        else:
            raise TypeError(
                "config_or_path must be a file path or a configuration dict."
            )

        if not isinstance(self.config, dict):
            raise ValueError(
                "Configuration must be a dictionary or YAML mapping."
            )

        paths_config = self.config.get("paths", {})
        landsat_config = self.config.get("landsat", {})
        temporal_config = self.config.get(
            "temporal_features",
            {}
        )
        temporal_context = temporal_config.get(
            "temporal_context",
            {}
        )

        self.landsat_dir = self._resolve_path(
            paths_config.get(
                "landsat_dir",
                "data/landsat"
            )
        )

        self.avhrr_dir = self._resolve_path(
            paths_config.get(
                "avhrr_dir",
                "data/avhrr"
            )
        )

        self.modis_dir = self._resolve_path(
            paths_config.get(
                "modis_dir",
                "data/modis"
            )
        )

        self.static_dir = self._resolve_path(
            paths_config.get(
                "static_dir",
                "data/static"
            )
        )

        self.era5_precip_dir = self._resolve_path(
            paths_config.get(
                "era5_precip_dir",
                "data/era5/precip"
            )
        )

        self.era5_temp_dir = self._resolve_path(
            paths_config.get(
                "era5_temp_dir",
                "data/era5/temp"
            )
        )

        configured_bands = tuple(
            landsat_config.get("bands", [])
        )

        if len(configured_bands) != EXPECTED_LANDSAT_BANDS:
            raise ValueError(
                f"Landsat configuration must contain exactly "
                f"{EXPECTED_LANDSAT_BANDS} bands. "
                f"Found {len(configured_bands)}."
            )

        if configured_bands != MODEL_BAND_ORDER:
            raise ValueError(
                "Landsat band order in config.yaml must be exactly: "
                f"{list(MODEL_BAND_ORDER)}. "
                f"Found: {list(configured_bands)}"
            )

        self.landsat_bands = configured_bands
        self.num_bands = EXPECTED_LANDSAT_BANDS

        data_years = landsat_config.get(
            "data_years",
            [1995, 2025]
        )

        if len(data_years) != 2:
            raise ValueError(
                "landsat.data_years must contain "
                "[start_year, end_year]."
            )

        self.data_start_year = int(data_years[0])
        self.data_end_year = int(data_years[1])

        if self.data_start_year > self.data_end_year:
            raise ValueError(
                "landsat.data_years start year cannot be greater than end year."
            )

        self.max_search_months = int(
            temporal_context.get(
                "max_search_months",
                12
            )
        )

        if self.max_search_months < 1:
            raise ValueError(
                "max_search_months must be at least 1."
            )

        preprocessing_config = self.config.get(
            "preprocessing",
            {}
        )

        target_grid_config = preprocessing_config.get(
            "target_grid",
            {}
        )

        self.target_resolution = float(
            target_grid_config.get(
                "resolution",
                30
            )
        )

        self._landsat_index: Optional[
            Dict[Tuple[int, int], Path]
        ] = None

        self._target_profile: Optional[dict] = None
        self._cached_dem: Optional[np.ndarray] = None

    def _resolve_path(
        self,
        configured_path: Union[str, Path]
    ) -> Path:
        """
        Resolve a configured path relative to the project root.
        """
        path = Path(configured_path)

        if path.is_absolute():
            return path

        return self.project_root / path

    @staticmethod
    def _validate_year_month(
        year: int,
        month: int
    ) -> Tuple[int, int]:
        """
        Validate and normalize a year/month pair.
        """
        year = int(year)
        month = int(month)

        if year < 1 or month < 1 or month > 12:
            raise ValueError(
                f"Invalid date specification: {year}-{month:02d}"
            )

        return year, month

    @staticmethod
    def _month_key_to_string(
        year: int,
        month: int
    ) -> str:
        """
        Convert a year/month pair into YYYY_MM format.
        """
        return f"{int(year)}_{int(month):02d}"

    def _find_landsat_file(
        self,
        year: int,
        month: int
    ) -> Optional[Path]:
        """
        Find the single stacked Landsat GeoTIFF for a month.

        Expected naming:
            landsat_YYYY_MM_lX.tif

        Examples:
            landsat_2023_01_l8.tif
            landsat_2012_03_l5.tif
        """
        year, month = self._validate_year_month(
            year,
            month
        )

        month_string = self._month_key_to_string(
            year,
            month
        )

        exact_pattern = f"landsat_{month_string}.tif"
        suffix_pattern = f"landsat_{month_string}_*.tif"

        exact_matches = sorted(
            self.landsat_dir.glob(exact_pattern)
        )

        if len(exact_matches) > 1:
            raise RuntimeError(
                f"Multiple Landsat files found for "
                f"{year}-{month:02d}: {exact_matches}"
            )

        if exact_matches:
            return exact_matches[0]

        suffix_matches = sorted(
            self.landsat_dir.glob(suffix_pattern)
        )

        if len(suffix_matches) > 1:
            raise RuntimeError(
                f"Multiple Landsat files found for "
                f"{year}-{month:02d}: {suffix_matches}. "
                "There must be exactly one stacked Landsat "
                "file per month."
            )

        if suffix_matches:
            return suffix_matches[0]

        return None

    def discover_landsat_files(
        self,
        force_refresh: bool = False
    ) -> Dict[Tuple[int, int], Path]:
        """
        Discover complete monthly Landsat observations.

        Each observation must be a seven-band GeoTIFF.

        Returns:
            {(year, month): stacked_landsat_path}
        """
        if (
            self._landsat_index is not None
            and not force_refresh
        ):
            return self._landsat_index

        index: Dict[Tuple[int, int], Path] = {}

        for year in range(
            self.data_start_year,
            self.data_end_year + 1
        ):
            for month in range(1, 13):

                path = self._find_landsat_file(
                    year,
                    month
                )

                if path is None:
                    continue

                try:
                    with rasterio.open(path) as src:

                        if src.count != EXPECTED_LANDSAT_BANDS:
                            raise ValueError(
                                f"Landsat file {path.name} contains "
                                f"{src.count} bands; expected "
                                f"{EXPECTED_LANDSAT_BANDS}."
                            )

                except rasterio.errors.RasterioIOError as exc:
                    raise RuntimeError(
                        f"Unable to open Landsat raster: {path}"
                    ) from exc

                index[(year, month)] = path

        self._landsat_index = index

        return index

    def get_available_months(self) -> List[Tuple[int, int]]:
        """
        Return all available complete Landsat observations
        in chronological order.
        """
        return sorted(
            self.discover_landsat_files().keys()
        )

    def detect_missing_months(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Identify months for which no complete Landsat observation exists.
        """
        start_year = (
            self.data_start_year
            if start_year is None
            else int(start_year)
        )

        end_year = (
            self.data_end_year
            if end_year is None
            else int(end_year)
        )

        if start_year > end_year:
            raise ValueError(
                "start_year cannot be greater than end_year."
            )

        index = self.discover_landsat_files()

        missing = []

        for year in range(
            start_year,
            end_year + 1
        ):
            for month in range(1, 13):

                if (year, month) not in index:
                    missing.append(
                        (year, month)
                    )

        return missing

    def get_reference_path(self) -> Path:
        """
        Return the raster used to define the target grid.

        The DEM is preferred. If the DEM is unavailable, the first
        available stacked Landsat raster is used.
        """
        dem_path = self.static_dir / "dem.tif"

        if dem_path.exists():
            return dem_path

        index = self.discover_landsat_files()

        if not index:
            raise FileNotFoundError(
                "No DEM and no Landsat raster are available "
                "to define the target grid."
            )

        first_key = sorted(
            index.keys()
        )[0]

        return index[first_key]

    def get_target_profile(self) -> dict:
        """
        Return and cache the common target raster profile.
        """
        if self._target_profile is not None:
            return self._target_profile

        reference_path = self.get_reference_path()

        with rasterio.open(reference_path) as src:

            self._target_profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "width": src.width,
                "height": src.height,
                "count": 1,
                "crs": src.crs,
                "transform": src.transform,
                "nodata": np.nan,
            }

        return self._target_profile

    def _read_aligned(
        self,
        path: Path,
        band_index: int = 1,
        resampling: Resampling = Resampling.bilinear
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read one raster band and align it to the common target grid.

        Source NoData values are converted to NaN before reprojection.

        Returns:
            aligned_array: Float32 array on the target grid.
            valid_mask: Boolean array indicating finite aligned pixels.
        """
        target_profile = self.get_target_profile()

        if band_index < 1:
            raise ValueError(
                f"Raster band index must be >= 1. "
                f"Got {band_index}."
            )

        with rasterio.open(path) as src:

            if band_index > src.count:
                raise ValueError(
                    f"Requested band {band_index} from "
                    f"{path.name}, but raster contains only "
                    f"{src.count} bands."
                )

            source = src.read(
                band_index
            ).astype(np.float32)

            source_nodata = src.nodata

            if source_nodata is not None:

                source[
                    np.isclose(
                        source,
                        source_nodata,
                        equal_nan=False
                    )
                ] = np.nan

            source[
                ~np.isfinite(source)
            ] = np.nan

            destination = np.full(
                (
                    target_profile["height"],
                    target_profile["width"]
                ),
                np.nan,
                dtype=np.float32
            )

            reproject(
                source=source,
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_profile["transform"],
                dst_crs=target_profile["crs"],
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=resampling
            )

        valid_mask = np.isfinite(
            destination
        )

        return destination, valid_mask

    def load_landsat_month(
        self,
        year: int,
        month: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a complete stacked Landsat observation.

        Source GeoTIFF order:
            1. Blue, 2. Green, 3. Red, 4. NIR, 5. SWIR1, 6. SWIR2, 7. Thermal

        Returned model order:
            1. Red, 2. Green, 3. Blue, 4. NIR, 5. SWIR1, 6. SWIR2, 7. Thermal

        Returns:
            bands: Array with shape (7, H, W).
            valid_mask: Boolean array with shape (H, W), true only where
                        all seven Landsat bands are valid.
        """
        year, month = self._validate_year_month(
            year,
            month
        )

        index = self.discover_landsat_files()

        path = index.get(
            (year, month)
        )

        if path is None:
            raise FileNotFoundError(
                f"No complete Landsat observation found "
                f"for {year}-{month:02d}."
            )

        with rasterio.open(path) as src:

            if src.count != EXPECTED_LANDSAT_BANDS:
                raise ValueError(
                    f"Landsat file {path.name} contains "
                    f"{src.count} bands; expected "
                    f"{EXPECTED_LANDSAT_BANDS}."
                )

        bands = []
        valid_masks = []

        for band_name in MODEL_BAND_ORDER:

            source_band_index = (
                SOURCE_TO_MODEL_INDEX[band_name]
            )

            band, valid_mask = self._read_aligned(
                path=path,
                band_index=source_band_index,
                resampling=Resampling.bilinear
            )

            bands.append(band)
            valid_masks.append(valid_mask)

        cube = np.stack(
            bands,
            axis=0
        )

        combined_valid_mask = np.logical_and.reduce(
            valid_masks
        )

        cube[
            :,
            ~combined_valid_mask
        ] = np.nan

        return (
            cube.astype(np.float32),
            combined_valid_mask
        )

    def _find_nearest_month(
        self,
        year: int,
        month: int,
        direction: int
    ) -> Optional[Tuple[int, int]]:
        """
        Find the nearest available Landsat observation before or after
        the requested month.

        direction:
            -1 = previous observation
            +1 = next observation
        """
        if direction not in (-1, 1):
            raise ValueError(
                "direction must be either -1 or +1."
            )

        index = self.discover_landsat_files()

        target_month_number = (
            year * 12 + month
        )

        for distance in range(
            1,
            self.max_search_months + 1
        ):

            candidate_number = (
                target_month_number
                + direction * distance
            )

            candidate_year = (
                candidate_number // 12
            )

            candidate_month = (
                candidate_number % 12
            )

            if candidate_month == 0:
                candidate_year -= 1
                candidate_month = 12

            if (
                candidate_year,
                candidate_month
            ) in index:

                return (
                    candidate_year,
                    candidate_month
                )

        return None

    def load_temporal_neighbors(
        self,
        year: int,
        month: int
    ) -> Dict[str, Any]:
        """
        Load the nearest previous and next Landsat observations.

        The returned dictionary uses the feature names expected by
        dataset.py:
            landsat_prev
            landsat_next
            dt_prev
            dt_next
            prev_available
            next_available

        Missing neighbours are represented by None and their
        corresponding availability flag is set to 0.0.
        """
        year, month = self._validate_year_month(
            year,
            month
        )

        previous_key = self._find_nearest_month(
            year=year,
            month=month,
            direction=-1
        )

        next_key = self._find_nearest_month(
            year=year,
            month=month,
            direction=1
        )

        previous_cube = None
        next_cube = None

        dt_prev = 0.0
        dt_next = 0.0

        prev_available = 0.0
        next_available = 0.0

        target_month_number = (
            year * 12 + month
        )

        if previous_key is not None:

            previous_cube, _ = (
                self.load_landsat_month(
                    previous_key[0],
                    previous_key[1]
                )
            )

            previous_month_number = (
                previous_key[0] * 12
                + previous_key[1]
            )

            dt_prev = float(
                target_month_number
                - previous_month_number
            )

            prev_available = 1.0

        if next_key is not None:

            next_cube, _ = (
                self.load_landsat_month(
                    next_key[0],
                    next_key[1]
                )
            )

            next_month_number = (
                next_key[0] * 12
                + next_key[1]
            )

            dt_next = float(
                next_month_number
                - target_month_number
            )

            next_available = 1.0

        return {
            "landsat_prev": previous_cube,
            "landsat_next": next_cube,
            "dt_prev": dt_prev,
            "dt_next": dt_next,
            "prev_available": prev_available,
            "next_available": next_available,
        }

    def _load_single_predictor(
        self,
        path: Path,
        resampling: Resampling = Resampling.bilinear
    ) -> np.ndarray:
        """
        Load and spatially align a single-band predictor.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Predictor raster not found: {path}"
            )

        array, _ = self._read_aligned(
            path=path,
            band_index=1,
            resampling=resampling
        )

        return array.astype(
            np.float32
        )

    def load_era5_precip(
        self,
        year: int,
        month: int
    ) -> np.ndarray:
        """
        Load monthly ERA5 precipitation.
        """
        year, month = self._validate_year_month(
            year,
            month
        )

        path = (
            self.era5_precip_dir
            / f"era5_precip_{year}_{month:02d}.tif"
        )

        return self._load_single_predictor(
            path
        )

    def load_era5_temp(
        self,
        year: int,
        month: int
    ) -> np.ndarray:
        """
        Load monthly ERA5 temperature.
        """
        year, month = self._validate_year_month(
            year,
            month
        )

        path = (
            self.era5_temp_dir
            / f"era5_temp_{year}_{month:02d}.tif"
        )

        return self._load_single_predictor(
            path
        )

    def load_era5_predictor(
        self,
        predictor_type: str,
        year: int,
        month: int
    ) -> np.ndarray:
        """
        Load an ERA5 predictor using the interface expected by dataset.py.

        Supported predictor types:
            precip
            temp
            temperature
        """
        predictor_type = str(
            predictor_type
        ).lower().strip()

        if predictor_type == "precip":
            return self.load_era5_precip(
                year,
                month
            )

        if predictor_type in (
            "temp",
            "temperature"
        ):
            return self.load_era5_temp(
                year,
                month
            )

        raise ValueError(
            f"Unknown ERA5 predictor type: {predictor_type}. "
            "Expected 'precip' or 'temp'."
        )

    def load_dem(self) -> np.ndarray:
        """
        Load and cache the static DEM.
        """
        if self._cached_dem is not None:
            return self._cached_dem

        path = self.static_dir / "dem.tif"

        if not path.exists():
            raise FileNotFoundError(
                f"DEM raster not found: {path}"
            )

        dem = self._load_single_predictor(
            path=path,
            resampling=Resampling.bilinear
        )

        self._cached_dem = dem.astype(
            np.float32
        )

        return self._cached_dem

    def load_static_dem(self) -> np.ndarray:
        """
        Load the static DEM using the interface expected by dataset.py.
        """
        return self.load_dem()

    def load_avhrr_ndvi(
        self,
        year: int,
        month: int
    ) -> np.ndarray:
        """
        Load AVHRR NDVI for independent evaluation.

        AVHRR is evaluation-only and is never used as an RBFN predictor.
        """
        year, month = self._validate_year_month(
            year,
            month
        )

        path = (
            self.avhrr_dir
            / f"avhrr_ndvi_{year}_{month:02d}.tif"
        )

        return self._load_single_predictor(
            path
        )

    def load_modis_ndvi(
        self,
        year: int,
        month: int
    ) -> np.ndarray:
        """
        Load MODIS NDVI for independent evaluation.
        """
        year, month = self._validate_year_month(
            year,
            month
        )

        path = (
            self.modis_dir
            / f"modis_ndvi_{year}_{month:02d}.tif"
        )

        return self._load_single_predictor(
            path
        )