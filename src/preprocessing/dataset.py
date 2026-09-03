# src/preprocessing/dataset.py
"""
Dataset construction, temporal context integration, and PyTorch streaming.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .raster_processor import RasterProcessor

PREVIOUS_BAND_START = 0
PREVIOUS_BAND_END = 7
NEXT_BAND_START = 7
NEXT_BAND_END = 14
DT_PREV_INDEX = 14
DT_NEXT_INDEX = 15
PREV_AVAILABLE_INDEX = 16
NEXT_AVAILABLE_INDEX = 17
ERA5_PRECIP_INDEX = 18
ERA5_TEMP_INDEX = 19
DEM_INDEX = 20

EXPECTED_IN_FEATURES = 21
EXPECTED_OUT_FEATURES = 7

ENVIRONMENTAL_FEATURE_INDICES = (
    ERA5_PRECIP_INDEX,
    ERA5_TEMP_INDEX,
    DEM_INDEX,
)


@dataclass
class MonthlySamples:
    """Pixel-level samples belonging to one monthly Landsat observation."""

    month_str: str
    X: np.ndarray
    Y: np.ndarray


@dataclass
class HoldoutSplitResult:
    """Chronologically separated Training, Validation, and Test samples."""

    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    train_months: List[str]
    val_months: List[str]
    test_months: List[str]


class RBFNInMemoryDataset(Dataset):
    """
    PyTorch Dataset for pixel-level RBFN samples held in memory.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array.")

        if Y.ndim != 2:
            raise ValueError("Y must be a 2-dimensional array.")

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must contain the same number of samples.")

        self.X = torch.as_tensor(
            np.ascontiguousarray(
                X,
                dtype=np.float32,
            )
        )

        self.Y = torch.as_tensor(
            np.ascontiguousarray(
                Y,
                dtype=np.float32,
            )
        )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


class GapFillDataset:
    """
    Constructs temporal Landsat gap-filling samples and PyTorch DataLoaders.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ) -> None:
        self.config = config
        self.processor = RasterProcessor(config)

        landsat_config = config.get("landsat", {})
        self.landsat_bands = landsat_config.get("bands", [])
        self.num_landsat_bands = len(self.landsat_bands)

        configured_num_bands = int(
            landsat_config.get(
                "num_bands",
                self.num_landsat_bands,
            )
        )

        if self.num_landsat_bands == 0:
            raise ValueError("No Landsat bands are configured.")

        if self.num_landsat_bands != configured_num_bands:
            raise ValueError(
                "Configured Landsat band count does not "
                "match the number of Landsat band names."
            )

        if self.num_landsat_bands != EXPECTED_OUT_FEATURES:
            raise ValueError(
                "The current RBFN pipeline requires exactly "
                f"{EXPECTED_OUT_FEATURES} Landsat target bands."
            )

        self._cached_dem: Optional[np.ndarray] = None

        model_config = config.get("model", {}).get("rbfn", {})
        pixels_per_month = model_config.get("pixels_per_month")

        if pixels_per_month is None:
            self.pixels_per_month: Optional[int] = None
        else:
            self.pixels_per_month = int(pixels_per_month)
            if self.pixels_per_month < 1:
                raise ValueError("pixels_per_month must be at least 1 when configured.")

        project_config = config.get("project", {})
        seed = int(project_config.get("seed", 42))

        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _month_to_string(
        year: int,
        month: int,
    ) -> str:
        return f"{int(year):04d}-{int(month):02d}"

    @staticmethod
    def _flatten_raster_features(
        array: np.ndarray,
    ) -> np.ndarray:
        """
        Convert raster arrays into pixel-feature matrices.

        (H, W)    -> (N, 1)
        (C, H, W) -> (N, C)
        """
        array = np.asarray(array)

        if array.ndim == 2:
            return np.ascontiguousarray(
                array.reshape(-1, 1),
                dtype=np.float32,
            )

        if array.ndim == 3:
            channels, height, width = array.shape
            return np.ascontiguousarray(
                array.reshape(channels, height * width).T,
                dtype=np.float32,
            )

        raise ValueError("Raster array must have shape (H, W) or (C, H, W).")

    @staticmethod
    def _validate_pixel_count(
        matrix: np.ndarray,
        expected_pixels: int,
        name: str,
    ) -> None:
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be a 2-dimensional pixel-feature matrix.")

        if matrix.shape[0] != expected_pixels:
            raise ValueError(
                f"{name} contains {matrix.shape[0]} pixels, expected {expected_pixels}."
            )

    def _load_monthly_predictors(
        self,
        year: int,
        month: int,
    ) -> np.ndarray:
        """
        Build the predictor matrix for one observed Landsat month.

        Feature order:
            0-6    Previous Landsat bands
            7-13   Next Landsat bands
            14     Previous temporal delta
            15     Next temporal delta
            16     Previous availability
            17     Next availability
            18     ERA5 precipitation
            19     ERA5 temperature
            20     DEM
        """
        neighbors = self.processor.load_temporal_neighbors(year, month)

        if self._cached_dem is None:
            self._cached_dem = self.processor.load_static_dem()

        dem_matrix = self._flatten_raster_features(self._cached_dem)
        spatial_pixels = dem_matrix.shape[0]

        if self.num_landsat_bands != EXPECTED_OUT_FEATURES:
            raise ValueError("Unexpected Landsat band count.")

        X = np.empty((spatial_pixels, EXPECTED_IN_FEATURES), dtype=np.float32)

        landsat_prev = neighbors.get("landsat_prev")
        if landsat_prev is not None:
            prev_matrix = self._flatten_raster_features(landsat_prev)
            self._validate_pixel_count(prev_matrix, spatial_pixels, "Previous Landsat")

            if prev_matrix.shape[1] != self.num_landsat_bands:
                raise ValueError("Previous Landsat observation has an unexpected number of bands.")

            X[:, PREVIOUS_BAND_START:PREVIOUS_BAND_END] = prev_matrix
            dt_prev = float(
                neighbors.get(
                    "dt_prev",
                    getattr(self.processor, "max_search_months", 12),
                )
            )
            prev_available = 1.0
        else:
            X[:, PREVIOUS_BAND_START:PREVIOUS_BAND_END] = 0.0
            dt_prev = float(getattr(self.processor, "max_search_months", 12))
            prev_available = 0.0

        landsat_next = neighbors.get("landsat_next")
        if landsat_next is not None:
            next_matrix = self._flatten_raster_features(landsat_next)
            self._validate_pixel_count(next_matrix, spatial_pixels, "Next Landsat")

            if next_matrix.shape[1] != self.num_landsat_bands:
                raise ValueError("Next Landsat observation has an unexpected number of bands.")

            X[:, NEXT_BAND_START:NEXT_BAND_END] = next_matrix
            dt_next = float(
                neighbors.get(
                    "dt_next",
                    getattr(self.processor, "max_search_months", 12),
                )
            )
            next_available = 1.0
        else:
            X[:, NEXT_BAND_START:NEXT_BAND_END] = 0.0
            dt_next = float(getattr(self.processor, "max_search_months", 12))
            next_available = 0.0

        X[:, DT_PREV_INDEX] = dt_prev
        X[:, DT_NEXT_INDEX] = dt_next
        X[:, PREV_AVAILABLE_INDEX] = prev_available
        X[:, NEXT_AVAILABLE_INDEX] = next_available

        precip = self.processor.load_era5_predictor("precip", year, month)
        precip_matrix = self._flatten_raster_features(precip)
        self._validate_pixel_count(precip_matrix, spatial_pixels, "ERA5 precipitation")

        if precip_matrix.shape[1] != 1:
            raise ValueError("ERA5 precipitation must contain exactly one feature.")

        X[:, ERA5_PRECIP_INDEX] = precip_matrix[:, 0]

        temperature = self.processor.load_era5_predictor("temp", year, month)
        temperature_matrix = self._flatten_raster_features(temperature)
        self._validate_pixel_count(temperature_matrix, spatial_pixels, "ERA5 temperature")

        if temperature_matrix.shape[1] != 1:
            raise ValueError("ERA5 temperature must contain exactly one feature.")

        X[:, ERA5_TEMP_INDEX] = temperature_matrix[:, 0]

        if dem_matrix.shape[1] != 1:
            raise ValueError("DEM must contain exactly one feature.")

        X[:, DEM_INDEX] = dem_matrix[:, 0]

        if X.shape[1] != EXPECTED_IN_FEATURES:
            raise RuntimeError(
                f"Internal feature construction error: expected {EXPECTED_IN_FEATURES} features, "
                f"constructed {X.shape[1]}."
            )

        return X

    def _load_monthly_targets(
        self,
        year: int,
        month: int,
    ) -> np.ndarray:
        """
        Load the complete Landsat observation used as target.
        """
        landsat_cube, _ = self.processor.load_landsat_month(year, month)
        Y = self._flatten_raster_features(landsat_cube)

        if Y.shape[1] != self.num_landsat_bands:
            raise ValueError(
                f"Unexpected Landsat target dimension: expected {self.num_landsat_bands}, "
                f"received {Y.shape[1]}."
            )

        if Y.shape[1] != EXPECTED_OUT_FEATURES:
            raise ValueError(
                f"Unexpected RBFN target dimension: expected {EXPECTED_OUT_FEATURES}, "
                f"received {Y.shape[1]}."
            )

        return Y

    @staticmethod
    def _create_valid_pixel_mask(
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        """
        Retain pixels with finite targets and environmental predictors.
        """
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must both be 2-dimensional.")

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must contain the same number of pixels.")

        if X.shape[1] != EXPECTED_IN_FEATURES:
            raise ValueError(f"X does not contain the expected {EXPECTED_IN_FEATURES} features.")

        if Y.shape[1] != EXPECTED_OUT_FEATURES:
            raise ValueError(f"Y does not contain the expected {EXPECTED_OUT_FEATURES} targets.")

        valid_targets = np.all(np.isfinite(Y), axis=1)

        valid_environment = np.all(
            np.isfinite(X[:, list(ENVIRONMENTAL_FEATURE_INDICES)]),
            axis=1,
        )

        valid_temporal_flags = np.isin(
            X[:, PREV_AVAILABLE_INDEX], [0.0, 1.0]
        ) & np.isin(X[:, NEXT_AVAILABLE_INDEX], [0.0, 1.0])

        return valid_targets & valid_environment & valid_temporal_flags

    def _get_available_training_months(
        self,
    ) -> List[Tuple[int, int]]:
        available_months = self.processor.get_available_months()

        if not available_months:
            raise ValueError("No complete Landsat observations were found.")

        training_years = self.config.get("landsat", {}).get("training_years", [])

        if len(training_years) != 2:
            raise ValueError("landsat.training_years must contain a start and end year.")

        training_start = int(training_years[0])
        training_end = int(training_years[1])

        if training_start > training_end:
            raise ValueError("landsat.training_years start year cannot exceed end year.")

        filtered_months = [
            (year, month)
            for year, month in available_months
            if training_start <= int(year) <= training_end
        ]

        if not filtered_months:
            raise ValueError(
                "No complete Landsat observations were found within the configured training period."
            )

        return sorted(filtered_months, key=lambda item: (item[0], item[1]))

    def _subsample_monthly_pixels(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        month_str: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly subsample valid pixels for one month.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"{month_str}: cannot subsample X and Y with different sample counts."
            )

        if self.pixels_per_month is None or X.shape[0] <= self.pixels_per_month:
            return X, Y

        selected_indices = self.rng.choice(
            X.shape[0],
            size=self.pixels_per_month,
            replace=False,
        )
        selected_indices = np.sort(selected_indices)

        return (
            np.ascontiguousarray(X[selected_indices], dtype=np.float32),
            np.ascontiguousarray(Y[selected_indices], dtype=np.float32),
        )

    def build_monthly_samples(
        self,
    ) -> List[MonthlySamples]:
        """
        Construct valid pixel samples for every complete Landsat month.
        """
        monthly_samples: List[MonthlySamples] = []
        available_months = self._get_available_training_months()

        for year, month in available_months:
            month_str = self._month_to_string(year, month)

            try:
                X = self._load_monthly_predictors(year, month)
                Y = self._load_monthly_targets(year, month)
            except (FileNotFoundError, ValueError) as error:
                print(f"Skipping {month_str}: {error}")
                continue

            if X.shape[0] != Y.shape[0]:
                raise ValueError(f"{month_str}: predictor and target pixel counts differ.")

            valid_mask = self._create_valid_pixel_mask(X, Y)

            X_clean = X[valid_mask]
            Y_clean = Y[valid_mask]

            if X_clean.shape[0] == 0:
                print(f"Skipping {month_str}: no valid pixels remain.")
                continue

            X_clean, Y_clean = self._subsample_monthly_pixels(
                X_clean,
                Y_clean,
                month_str,
            )

            monthly_samples.append(
                MonthlySamples(
                    month_str=month_str,
                    X=X_clean,
                    Y=Y_clean,
                )
            )

            print(
                f"Built {month_str}: {X_clean.shape[0]} pixels | "
                f"{X_clean.shape[1]} features | {Y_clean.shape[1]} targets"
            )

        if not monthly_samples:
            raise ValueError("No valid monthly samples could be constructed.")

        return sorted(monthly_samples, key=lambda sample: sample.month_str)

    def create_temporal_holdout(
        self,
        monthly_samples: List[MonthlySamples],
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
    ) -> HoldoutSplitResult:
        """
        Perform strict chronological monthly splitting.
        """
        if not monthly_samples:
            raise ValueError("monthly_samples cannot be empty.")

        rbfn_config = self.config.get("model", {}).get("rbfn", {})

        if val_ratio is None:
            val_ratio = float(rbfn_config.get("validation_holdout_ratio", 0.22))

        if test_ratio is None:
            test_ratio = float(rbfn_config.get("test_holdout_ratio", 0.15))

        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio must be between 0 and 1.")

        if not (0.0 < test_ratio < 1.0):
            raise ValueError("test_ratio must be between 0 and 1.")

        if val_ratio + test_ratio >= 1.0:
            raise ValueError("val_ratio + test_ratio must be less than 1.")

        sorted_samples = sorted(monthly_samples, key=lambda sample: sample.month_str)
        n_total = len(sorted_samples)

        if n_total < 3:
            raise ValueError("At least 3 monthly observations are required for splitting.")

        n_test = max(1, int(np.round(n_total * test_ratio)))
        n_val = max(1, int(np.round(n_total * val_ratio)))
        n_train = n_total - n_val - n_test

        if n_train < 1:
            raise ValueError("Insufficient observations for the Training partition.")

        train_samples = sorted_samples[:n_train]
        val_samples = sorted_samples[n_train : n_train + n_val]
        test_samples = sorted_samples[n_train + n_val :]

        if not train_samples:
            raise ValueError("Training partition is empty.")
        if not val_samples:
            raise ValueError("Validation partition is empty.")
        if not test_samples:
            raise ValueError("Test partition is empty.")

        X_train = np.vstack([sample.X for sample in train_samples])
        Y_train = np.vstack([sample.Y for sample in train_samples])

        X_val = np.vstack([sample.X for sample in val_samples])
        Y_val = np.vstack([sample.Y for sample in val_samples])

        X_test = np.vstack([sample.X for sample in test_samples])
        Y_test = np.vstack([sample.Y for sample in test_samples])

        feature_dimension = X_train.shape[1]
        target_dimension = Y_train.shape[1]

        if feature_dimension != EXPECTED_IN_FEATURES:
            raise ValueError(
                f"Training feature dimension error: expected {EXPECTED_IN_FEATURES}, "
                f"received {feature_dimension}."
            )

        if target_dimension != EXPECTED_OUT_FEATURES:
            raise ValueError(
                f"Training target dimension error: expected {EXPECTED_OUT_FEATURES}, "
                f"received {target_dimension}."
            )

        for name, X, Y in [
            ("Training", X_train, Y_train),
            ("Validation", X_val, Y_val),
            ("Test", X_test, Y_test),
        ]:
            if X.ndim != 2 or Y.ndim != 2:
                raise ValueError(f"{name} matrices must be 2-dimensional.")

            if X.shape[0] != Y.shape[0]:
                raise ValueError(f"{name} predictor and target sample counts do not match.")

            if X.shape[1] != feature_dimension:
                raise ValueError(f"{name} feature dimension differs from Training.")

            if Y.shape[1] != target_dimension:
                raise ValueError(f"{name} target dimension differs from Training.")

        return HoldoutSplitResult(
            X_train=np.ascontiguousarray(X_train, dtype=np.float32),
            Y_train=np.ascontiguousarray(Y_train, dtype=np.float32),
            X_val=np.ascontiguousarray(X_val, dtype=np.float32),
            Y_val=np.ascontiguousarray(Y_val, dtype=np.float32),
            X_test=np.ascontiguousarray(X_test, dtype=np.float32),
            Y_test=np.ascontiguousarray(Y_test, dtype=np.float32),
            train_months=[sample.month_str for sample in train_samples],
            val_months=[sample.month_str for sample in val_samples],
            test_months=[sample.month_str for sample in test_samples],
        )

    def create_rbfn_dataloaders(
        self,
        batch_size: int = 4096,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, HoldoutSplitResult]:
        """
        Build chronological RBFN DataLoaders.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        samples = self.build_monthly_samples()

        split = self.create_temporal_holdout(
            monthly_samples=samples,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        train_dataset = RBFNInMemoryDataset(split.X_train, split.Y_train)
        val_dataset = RBFNInMemoryDataset(split.X_val, split.Y_val)
        test_dataset = RBFNInMemoryDataset(split.X_test, split.Y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader, split