# src/preprocessing/raster_processor.py

"""
Central raster discovery, loading and spatial alignment utilities.

This module is responsible for:

- Resolving project paths from config.yaml
- Discovering complete Landsat monthly observations
- Detecting missing Landsat months
- Defining a common target grid
- Reprojecting and resampling rasters to that grid
- Converting NoData values to NaN
- Loading complete Landsat monthly raster cubes
- Loading ERA5 precipitation and temperature predictors
- Loading the static DEM
- Retrieving temporal Landsat neighbours
- Loading AVHRR and MODIS NDVI evaluation data

All rasters returned by this processor are aligned to the same target grid.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


class RasterProcessor:
    """
    Central raster-processing component for the Landsat temporal gap-filling
    pipeline.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ) -> None:
        """
        Initialize paths, Landsat configuration and spatial settings.
        """

        self.config = config

        paths_config = config.get(
            "paths",
            {},
        )

        self.data_dir = Path(
            paths_config.get(
                "data_dir",
                "data",
            )
        )

        self.landsat_dir = Path(
            paths_config.get(
                "landsat_dir",
                self.data_dir / "landsat",
            )
        )

        self.avhrr_dir = Path(
            paths_config.get(
                "avhrr_dir",
                self.data_dir / "avhrr",
            )
        )

        self.modis_dir = Path(
            paths_config.get(
                "modis_dir",
                self.data_dir / "modis",
            )
        )

        self.static_dir = Path(
            paths_config.get(
                "static_dir",
                self.data_dir / "static",
            )
        )

        self.era5_precip_dir = Path(
            paths_config.get(
                "era5_precip_dir",
                self.data_dir / "era5" / "precip",
            )
        )

        self.era5_temp_dir = Path(
            paths_config.get(
                "era5_temp_dir",
                self.data_dir / "era5" / "temp",
            )
        )

        self.landsat_bands = list(
            config.get(
                "landsat",
                {},
            ).get(
                "bands",
                [],
            )
        )

        if not self.landsat_bands:
            raise ValueError(
                "No Landsat bands are defined in config.yaml."
            )

        landsat_config = config.get(
            "landsat",
            {},
        )

        self.data_years = (
            int(
                landsat_config.get(
                    "data_years",
                    [1995, 2025],
                )[0]
            ),
            int(
                landsat_config.get(
                    "data_years",
                    [1995, 2025],
                )[1]
            ),
        )

        temporal_config = config.get(
            "temporal_features",
            {},
        ).get(
            "temporal_context",
            {},
        )

        self.max_search_months = int(
            temporal_config.get(
                "max_search_months",
                12,
            )
        )

        preprocessing_config = config.get(
            "preprocessing",
            {},
        )

        self.spatial_alignment = bool(
            preprocessing_config.get(
                "spatial_alignment",
                True,
            )
        )

        self._target_profile: Optional[
            Dict[str, Any]
        ] = None

        self._landsat_index: Optional[
            Dict[
                Tuple[int, int],
                Dict[str, Path],
            ]
        ] = None

    @staticmethod
    def _month_string(
        year: int,
        month: int,
    ) -> str:
        """
        Convert year and month to YYYY-MM format.
        """

        return (
            f"{int(year):04d}-"
            f"{int(month):02d}"
        )

    @staticmethod
    def _validate_month(
        year: int,
        month: int,
    ) -> None:
        """
        Validate temporal identifiers.
        """

        if month < 1 or month > 12:
            raise ValueError(
                f"Invalid month: {month}. "
                "Month must be between 1 and 12."
            )

        if year < 1:
            raise ValueError(
                f"Invalid year: {year}."
            )

    def _find_band_file(
        self,
        band: str,
        year: int,
        month: int,
    ) -> Optional[Path]:
        """
        Find a Landsat raster file for one band and month.

        The expected naming structure contains:

            band_year_month

        Examples:

            Red_2001_05.tif
            NIR_2001_05_scene.tif

        Additional filename components after the month are permitted.
        """

        month_string = (
            f"{int(month):02d}"
        )

        patterns = [
            f"{band}_{year}_{month_string}.tif",
            f"{band}_{year}_{month_string}_*.tif",
        ]

        for pattern in patterns:
            matches = sorted(
                self.landsat_dir.glob(pattern)
            )

            if matches:
                return matches[0]

        return None

    def discover_landsat_files(
        self,
    ) -> Dict[
        Tuple[int, int],
        Dict[str, Path],
    ]:
        """
        Discover complete Landsat monthly observations.

        A month is considered available only when every configured Landsat band
        is present.

        Returns
        -------
        dict
            Mapping:

                (year, month)
                    ->
                {band_name: raster_path}
        """

        if self._landsat_index is not None:
            return self._landsat_index

        if not self.landsat_dir.exists():
            raise FileNotFoundError(
                "Landsat directory does not exist: "
                f"{self.landsat_dir}"
            )

        start_year, end_year = self.data_years

        index: Dict[
            Tuple[int, int],
            Dict[str, Path],
        ] = {}

        for year in range(
            start_year,
            end_year + 1,
        ):
            for month in range(1, 13):
                band_paths: Dict[str, Path] = {}
                complete = True

                for band in self.landsat_bands:
                    path = self._find_band_file(
                        band=band,
                        year=year,
                        month=month,
                    )

                    if path is None:
                        complete = False
                        break

                    band_paths[band] = path

                if complete:
                    index[(year, month)] = band_paths

        self._landsat_index = index
        return index

    def get_available_months(
        self,
    ) -> List[Tuple[int, int]]:
        """
        Return chronologically sorted complete Landsat observations.
        """

        index = self.discover_landsat_files()

        return sorted(
            index.keys(),
            key=lambda item: (
                item[0],
                item[1],
            ),
        )

    def detect_missing_months(
        self,
    ) -> List[Tuple[int, int]]:
        """
        Detect months without complete Landsat observations.
        """

        available = set(
            self.get_available_months()
        )

        start_year, end_year = self.data_years

        missing: List[Tuple[int, int]] = []

        for year in range(
            start_year,
            end_year + 1,
        ):
            for month in range(1, 13):
                key = (year, month)
                if key not in available:
                    missing.append(key)

        return missing

    def get_reference_path(self) -> Path:
        """
        Select the raster used to define the common target grid.

        The DEM is preferred because it is static and provides a consistent
        spatial reference across the entire project.
        """

        dem_path = self.static_dir / "dem.tif"

        if dem_path.exists():
            return dem_path

        available_months = self.get_available_months()

        if not available_months:
            raise FileNotFoundError(
                "Unable to determine a reference grid because no complete "
                "Landsat observations or DEM were found."
            )

        first_year, first_month = available_months[0]
        index = self.discover_landsat_files()
        first_band = self.landsat_bands[0]

        return index[(first_year, first_month)][first_band]

    def get_target_profile(self) -> Dict[str, Any]:
        """
        Return the cached common target raster profile.
        """

        if self._target_profile is not None:
            return self._target_profile

        reference_path = self.get_reference_path()

        with rasterio.open(reference_path) as source:
            profile = {
                "crs": source.crs,
                "transform": source.transform,
                "width": source.width,
                "height": source.height,
            }

        if profile["crs"] is None:
            raise ValueError(
                "Reference raster has no CRS: "
                f"{reference_path}"
            )

        self._target_profile = profile
        return profile

    def _read_aligned(
        self,
        raster_path: Path,
        resampling: Resampling = Resampling.bilinear,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a raster and align it to the common target grid.

        NoData values are converted to NaN before reprojection.

        Returns
        -------
        tuple
            aligned_array
                Raster values on the target grid.
            valid_mask
                Boolean mask indicating finite pixels.
        """

        raster_path = Path(raster_path)

        if not raster_path.exists():
            raise FileNotFoundError(
                f"Raster file not found: {raster_path}"
            )

        target = self.get_target_profile()

        with rasterio.open(raster_path) as source:
            source_array = source.read(1).astype(np.float32)
            source_nodata = source.nodata

            if source_nodata is not None:
                source_array[source_array == source_nodata] = np.nan

            destination = np.full(
                (
                    target["height"],
                    target["width"],
                ),
                np.nan,
                dtype=np.float32,
            )

            if (
                source.crs == target["crs"]
                and source.transform == target["transform"]
                and source.width == target["width"]
                and source.height == target["height"]
            ):
                destination = source_array
            else:
                reproject(
                    source=source_array,
                    destination=destination,
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=np.nan,
                    dst_transform=target["transform"],
                    dst_crs=target["crs"],
                    dst_nodata=np.nan,
                    resampling=resampling,
                )

        valid_mask = np.isfinite(destination)

        return destination, valid_mask

    def load_landsat_month(
        self,
        year: int,
        month: int,
    ) -> np.ndarray:
        """
        Load all configured Landsat bands for one complete monthly observation.

        Returns
        -------
        np.ndarray
            Raster cube with shape: (number_of_bands, height, width)
        """

        self._validate_month(year, month)

        index = self.discover_landsat_files()
        key = (int(year), int(month))

        if key not in index:
            raise FileNotFoundError(
                "Complete Landsat observation is unavailable for "
                f"{self._month_string(year, month)}."
            )

        aligned_bands: List[np.ndarray] = []

        for band in self.landsat_bands:
            band_path = index[key][band]

            aligned_band, _ = self._read_aligned(
                band_path,
                resampling=Resampling.bilinear,
            )

            aligned_bands.append(aligned_band)

        return np.stack(aligned_bands, axis=0).astype(np.float32)

    def load_era5_predictor(
        self,
        variable: str,
        year: int,
        month: int,
    ) -> np.ndarray:
        """
        Load an ERA5 predictor aligned to the common target grid.

        Supported variables:
            precip
            temp

        Expected filenames:
            era5_precip_YYYY_MM.tif
            era5_temp_YYYY_MM.tif
        """

        self._validate_month(year, month)
        month_string = f"{int(month):02d}"
        variable = variable.lower()

        if variable == "precip":
            raster_path = (
                self.era5_precip_dir
                / f"era5_precip_{year}_{month_string}.tif"
            )
        elif variable == "temp":
            raster_path = (
                self.era5_temp_dir
                / f"era5_temp_{year}_{month_string}.tif"
            )
        else:
            raise ValueError(
                f"Unsupported ERA5 variable: {variable}. "
                "Supported variables are 'precip' and 'temp'."
            )

        aligned_array, _ = self._read_aligned(
            raster_path,
            resampling=Resampling.bilinear,
        )

        return aligned_array

    def load_static_dem(self) -> np.ndarray:
        """
        Load the static DEM aligned to the common target grid.
        """

        dem_path = self.static_dir / "dem.tif"

        aligned_array, _ = self._read_aligned(
            dem_path,
            resampling=Resampling.bilinear,
        )

        return aligned_array

    def _month_distance(
        self,
        year_a: int,
        month_a: int,
        year_b: int,
        month_b: int,
    ) -> int:
        """
        Calculate absolute temporal distance between two months.
        """

        index_a = int(year_a) * 12 + int(month_a)
        index_b = int(year_b) * 12 + int(month_b)

        return abs(index_a - index_b)

    def _find_previous_observation(
        self,
        year: int,
        month: int,
    ) -> Optional[Tuple[int, int]]:
        """
        Find the nearest complete Landsat observation before a target month.
        """

        available = self.get_available_months()
        target_index = int(year) * 12 + int(month)
        candidates = []

        for obs_year, obs_month in available:
            observation_index = obs_year * 12 + obs_month

            if observation_index < target_index:
                distance = target_index - observation_index

                if distance <= self.max_search_months:
                    candidates.append(
                        (distance, obs_year, obs_month)
                    )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        _, nearest_year, nearest_month = candidates[0]

        return nearest_year, nearest_month

    def _find_next_observation(
        self,
        year: int,
        month: int,
    ) -> Optional[Tuple[int, int]]:
        """
        Find the nearest complete Landsat observation after a target month.
        """

        available = self.get_available_months()
        target_index = int(year) * 12 + int(month)
        candidates = []

        for obs_year, obs_month in available:
            observation_index = obs_year * 12 + obs_month

            if observation_index > target_index:
                distance = observation_index - target_index

                if distance <= self.max_search_months:
                    candidates.append(
                        (distance, obs_year, obs_month)
                    )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        _, nearest_year, nearest_month = candidates[0]

        return nearest_year, nearest_month

    def load_temporal_neighbors(
        self,
        year: int,
        month: int,
    ) -> Dict[str, Any]:
        """
        Load the nearest previous and next complete Landsat observations.

        This method is intended for the temporal-context gap-filling stage.

        The returned dictionary contains:
            landsat_prev
            landsat_next
            dt_prev
            dt_next
            prev_date
            next_date
        """

        self._validate_month(year, month)

        previous = self._find_previous_observation(year, month)
        next_observation = self._find_next_observation(year, month)

        previous_cube = None
        next_cube = None
        dt_prev = None
        dt_next = None

        if previous is not None:
            previous_year, previous_month = previous
            previous_cube = self.load_landsat_month(
                previous_year,
                previous_month,
            )
            dt_prev = self._month_distance(
                year,
                month,
                previous_year,
                previous_month,
            )

        if next_observation is not None:
            next_year, next_month = next_observation
            next_cube = self.load_landsat_month(
                next_year,
                next_month,
            )
            dt_next = self._month_distance(
                year,
                month,
                next_year,
                next_month,
            )

        return {
            "landsat_prev": previous_cube,
            "landsat_next": next_cube,
            "dt_prev": dt_prev,
            "dt_next": dt_next,
            "prev_date": previous,
            "next_date": next_observation,
        }

    def load_avhrr_ndvi(
        self,
        year: int,
        month: int,
    ) -> np.ndarray:
        """
        Load AVHRR NDVI for independent gap-period evaluation.

        Expected filename:
            avhrr_ndvi_YYYY_MM.tif
        """

        self._validate_month(year, month)

        raster_path = (
            self.avhrr_dir
            / f"avhrr_ndvi_{year}_{month:02d}.tif"
        )

        aligned_array, _ = self._read_aligned(
            raster_path,
            resampling=Resampling.bilinear,
        )

        return aligned_array

    def load_modis_ndvi(
        self,
        year: int,
        month: int,
    ) -> np.ndarray:
        """
        Load MODIS NDVI for independent evaluation.

        Expected filename:
            modis_ndvi_YYYY_MM.tif
        """

        self._validate_month(year, month)

        raster_path = (
            self.modis_dir
            / f"modis_ndvi_{year}_{month:02d}.tif"
        )

        aligned_array, _ = self._read_aligned(
            raster_path,
            resampling=Resampling.bilinear,
        )

        return aligned_array