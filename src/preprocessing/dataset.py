# src/preprocessing/dataset.py

"""
Dataset construction, temporal context integration, and PyTorch streaming.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .raster_processor import RasterProcessor


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
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        if Y.ndim != 2:
            raise ValueError(
                "Y must be a 2-dimensional array."
            )

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                "X and Y must contain the same number of samples."
            )

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

        self.processor = RasterProcessor(
            config
        )

        landsat_config = config.get(
            "landsat",
            {}
        )

        self.landsat_bands = landsat_config.get(
            "bands",
            []
        )

        self.num_landsat_bands = len(
            self.landsat_bands
        )

        if self.num_landsat_bands == 0:
            raise ValueError(
                "No Landsat bands are configured."
            )

        self._cached_dem = None

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

        array = np.asarray(
            array
        )

        if array.ndim == 2:

            return np.ascontiguousarray(
                array.reshape(
                    -1,
                    1,
                ),
                dtype=np.float32,
            )

        if array.ndim == 3:

            channels, height, width = (
                array.shape
            )

            return np.ascontiguousarray(
                array.reshape(
                    channels,
                    height * width,
                ).T,
                dtype=np.float32,
            )

        raise ValueError(
            "Raster array must have shape "
            "(H, W) or (C, H, W)."
        )

    @staticmethod
    def _validate_pixel_count(
        matrix: np.ndarray,
        expected_pixels: int,
        name: str,
    ) -> None:

        if matrix.ndim != 2:
            raise ValueError(
                f"{name} must be a 2-dimensional "
                "pixel-feature matrix."
            )

        if matrix.shape[0] != expected_pixels:
            raise ValueError(
                f"{name} contains {matrix.shape[0]} "
                f"pixels, expected {expected_pixels}."
            )

    def _load_monthly_predictors(
        self,
        year: int,
        month: int,
    ) -> np.ndarray:
        """
        Build the predictor matrix for one observed Landsat month.

        Feature order:

            Previous Landsat bands
            Next Landsat bands
            Previous temporal delta
            Next temporal delta
            Previous availability
            Next availability
            ERA5 precipitation
            ERA5 temperature
            DEM

        Missing temporal Landsat observations are represented by finite
        zero values and identified explicitly by their availability flags.
        """

        neighbors = (
            self.processor.load_temporal_neighbors(
                year,
                month,
            )
        )

        if self._cached_dem is None:

            self._cached_dem = (
                self.processor.load_static_dem()
            )

        dem_matrix = (
            self._flatten_raster_features(
                self._cached_dem
            )
        )

        spatial_pixels = (
            dem_matrix.shape[0]
        )

        total_features = (
            2 * self.num_landsat_bands
            + 7
        )

        X = np.empty(
            (
                spatial_pixels,
                total_features,
            ),
            dtype=np.float32,
        )

        idx = 0

        max_search_months = float(
            getattr(
                self.processor,
                "max_search_months",
                12,
            )
        )

        landsat_prev = neighbors.get(
            "landsat_prev"
        )

        if landsat_prev is not None:

            prev_matrix = (
                self._flatten_raster_features(
                    landsat_prev
                )
            )

            self._validate_pixel_count(
                prev_matrix,
                spatial_pixels,
                "Previous Landsat",
            )

            if (
                prev_matrix.shape[1]
                != self.num_landsat_bands
            ):
                raise ValueError(
                    "Previous Landsat observation "
                    "has an unexpected number of bands."
                )

            X[
                :,
                idx:idx + self.num_landsat_bands
            ] = prev_matrix

            dt_prev = float(
                neighbors.get(
                    "dt_prev",
                    max_search_months,
                )
            )

            prev_available = 1.0

        else:

            X[
                :,
                idx:idx + self.num_landsat_bands
            ] = 0.0

            dt_prev = max_search_months
            prev_available = 0.0

        idx += self.num_landsat_bands

        landsat_next = neighbors.get(
            "landsat_next"
        )

        if landsat_next is not None:

            next_matrix = (
                self._flatten_raster_features(
                    landsat_next
                )
            )

            self._validate_pixel_count(
                next_matrix,
                spatial_pixels,
                "Next Landsat",
            )

            if (
                next_matrix.shape[1]
                != self.num_landsat_bands
            ):
                raise ValueError(
                    "Next Landsat observation "
                    "has an unexpected number of bands."
                )

            X[
                :,
                idx:idx + self.num_landsat_bands
            ] = next_matrix

            dt_next = float(
                neighbors.get(
                    "dt_next",
                    max_search_months,
                )
            )

            next_available = 1.0

        else:

            X[
                :,
                idx:idx + self.num_landsat_bands
            ] = 0.0

            dt_next = max_search_months
            next_available = 0.0

        idx += self.num_landsat_bands

        X[
            :,
            idx
        ] = dt_prev

        idx += 1

        X[
            :,
            idx
        ] = dt_next

        idx += 1

        X[
            :,
            idx
        ] = prev_available

        idx += 1

        X[
            :,
            idx
        ] = next_available

        idx += 1

        precip = (
            self.processor.load_era5_predictor(
                "precip",
                year,
                month,
            )
        )

        precip_matrix = (
            self._flatten_raster_features(
                precip
            )
        )

        self._validate_pixel_count(
            precip_matrix,
            spatial_pixels,
            "ERA5 precipitation",
        )

        if precip_matrix.shape[1] != 1:
            raise ValueError(
                "ERA5 precipitation must contain "
                "exactly one feature."
            )

        X[
            :,
            idx
        ] = precip_matrix[
            :,
            0
        ]

        idx += 1

        temperature = (
            self.processor.load_era5_predictor(
                "temp",
                year,
                month,
            )
        )

        temperature_matrix = (
            self._flatten_raster_features(
                temperature
            )
        )

        self._validate_pixel_count(
            temperature_matrix,
            spatial_pixels,
            "ERA5 temperature",
        )

        if temperature_matrix.shape[1] != 1:
            raise ValueError(
                "ERA5 temperature must contain "
                "exactly one feature."
            )

        X[
            :,
            idx
        ] = temperature_matrix[
            :,
            0
        ]

        idx += 1

        if dem_matrix.shape[1] != 1:
            raise ValueError(
                "DEM must contain exactly one feature."
            )

        X[
            :,
            idx
        ] = dem_matrix[
            :,
            0
        ]

        idx += 1

        if idx != total_features:
            raise RuntimeError(
                "Internal feature construction error: "
                f"expected {total_features} features, "
                f"constructed {idx}."
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

        landsat_cube = (
            self.processor.load_landsat_month(
                year,
                month,
            )
        )

        Y = (
            self._flatten_raster_features(
                landsat_cube
            )
        )

        if Y.shape[1] != self.num_landsat_bands:

            raise ValueError(
                "Unexpected Landsat target dimension: "
                f"expected {self.num_landsat_bands}, "
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

        Missing temporal Landsat neighbours are represented by finite zero
        values and are therefore not removed by this mask. Their availability
        indicators are retained as model inputs.
        """

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError(
                "X and Y must both be 2-dimensional."
            )

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                "X and Y must contain the same "
                "number of pixels."
            )

        valid_targets = np.all(
            np.isfinite(Y),
            axis=1,
        )

        valid_environment = np.all(
            np.isfinite(
                X[:, -3:]
            ),
            axis=1,
        )

        return (
            valid_targets
            & valid_environment
        )

    def _get_available_training_months(
        self,
    ) -> List[Tuple[int, int]]:

        available_months = (
            self.processor.get_available_months()
        )

        if not available_months:
            raise ValueError(
                "No complete Landsat observations were found."
            )

        return sorted(
            available_months,
            key=lambda item: (
                item[0],
                item[1],
            ),
        )

    def build_monthly_samples(
        self,
    ) -> List[MonthlySamples]:
        """
        Construct valid pixel samples for every complete Landsat month.
        """

        monthly_samples: List[
            MonthlySamples
        ] = []

        available_months = (
            self._get_available_training_months()
        )

        for year, month in available_months:

            month_str = (
                self._month_to_string(
                    year,
                    month,
                )
            )

            try:

                X = (
                    self._load_monthly_predictors(
                        year,
                        month,
                    )
                )

                Y = (
                    self._load_monthly_targets(
                        year,
                        month,
                    )
                )

            except (
                FileNotFoundError,
                ValueError,
            ) as error:

                print(
                    f"Skipping {month_str}: {error}"
                )

                continue

            if X.shape[0] != Y.shape[0]:

                raise ValueError(
                    f"{month_str}: predictor and "
                    "target pixel counts differ."
                )

            valid_mask = (
                self._create_valid_pixel_mask(
                    X,
                    Y,
                )
            )

            X_clean = X[
                valid_mask
            ]

            Y_clean = Y[
                valid_mask
            ]

            if X_clean.shape[0] == 0:

                print(
                    f"Skipping {month_str}: "
                    "no valid pixels remain."
                )

                continue

            monthly_samples.append(
                MonthlySamples(
                    month_str=month_str,
                    X=X_clean,
                    Y=Y_clean,
                )
            )

            print(
                f"Built {month_str}: "
                f"{X_clean.shape[0]} pixels | "
                f"{X_clean.shape[1]} features | "
                f"{Y_clean.shape[1]} targets"
            )

        if not monthly_samples:

            raise ValueError(
                "No valid monthly samples could be constructed."
            )

        return sorted(
            monthly_samples,
            key=lambda sample: sample.month_str,
        )

    def create_temporal_holdout(
        self,
        monthly_samples: List[MonthlySamples],
        val_ratio: float = 0.20,
        test_ratio: float = 0.15,
    ) -> HoldoutSplitResult:
        """
        Perform strict chronological monthly splitting.

        Earlier observations are assigned to Training, followed by
        Validation, and finally Test.
        """

        if not monthly_samples:
            raise ValueError(
                "monthly_samples cannot be empty."
            )

        if not (
            0.0 < val_ratio < 1.0
        ):
            raise ValueError(
                "val_ratio must be between 0 and 1."
            )

        if not (
            0.0 < test_ratio < 1.0
        ):
            raise ValueError(
                "test_ratio must be between 0 and 1."
            )

        if (
            val_ratio + test_ratio
            >= 1.0
        ):
            raise ValueError(
                "val_ratio + test_ratio must "
                "be less than 1."
            )

        sorted_samples = sorted(
            monthly_samples,
            key=lambda sample: sample.month_str,
        )

        n_total = len(
            sorted_samples
        )

        if n_total < 3:
            raise ValueError(
                "At least 3 monthly observations "
                "are required for splitting."
            )

        n_test = max(
            1,
            int(
                np.round(
                    n_total * test_ratio
                )
            ),
        )

        n_val = max(
            1,
            int(
                np.round(
                    n_total * val_ratio
                )
            ),
        )

        n_train = (
            n_total
            - n_val
            - n_test
        )

        if n_train < 1:
            raise ValueError(
                "Insufficient observations for "
                "the Training partition."
            )

        train_samples = sorted_samples[
            :n_train
        ]

        val_samples = sorted_samples[
            n_train:n_train + n_val
        ]

        test_samples = sorted_samples[
            n_train + n_val:
        ]

        if not train_samples:
            raise ValueError(
                "Training partition is empty."
            )

        if not val_samples:
            raise ValueError(
                "Validation partition is empty."
            )

        if not test_samples:
            raise ValueError(
                "Test partition is empty."
            )

        X_train = np.vstack(
            [
                sample.X
                for sample in train_samples
            ]
        )

        Y_train = np.vstack(
            [
                sample.Y
                for sample in train_samples
            ]
        )

        X_val = np.vstack(
            [
                sample.X
                for sample in val_samples
            ]
        )

        Y_val = np.vstack(
            [
                sample.Y
                for sample in val_samples
            ]
        )

        X_test = np.vstack(
            [
                sample.X
                for sample in test_samples
            ]
        )

        Y_test = np.vstack(
            [
                sample.Y
                for sample in test_samples
            ]
        )

        feature_dimension = (
            X_train.shape[1]
        )

        target_dimension = (
            Y_train.shape[1]
        )

        for name, X, Y in [
            (
                "Training",
                X_train,
                Y_train,
            ),
            (
                "Validation",
                X_val,
                Y_val,
            ),
            (
                "Test",
                X_test,
                Y_test,
            ),
        ]:

            if X.ndim != 2 or Y.ndim != 2:
                raise ValueError(
                    f"{name} matrices must be 2-dimensional."
                )

            if X.shape[0] != Y.shape[0]:
                raise ValueError(
                    f"{name} predictor and target "
                    "sample counts do not match."
                )

            if X.shape[1] != feature_dimension:
                raise ValueError(
                    f"{name} feature dimension differs "
                    "from Training."
                )

            if Y.shape[1] != target_dimension:
                raise ValueError(
                    f"{name} target dimension differs "
                    "from Training."
                )

        return HoldoutSplitResult(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            train_months=[
                sample.month_str
                for sample in train_samples
            ],
            val_months=[
                sample.month_str
                for sample in val_samples
            ],
            test_months=[
                sample.month_str
                for sample in test_samples
            ],
        )

    def create_rbfn_dataloaders(
        self,
        batch_size: int = 4096,
        val_ratio: float = 0.20,
        test_ratio: float = 0.15,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> Tuple[
        DataLoader,
        DataLoader,
        DataLoader,
        HoldoutSplitResult,
    ]:
        """
        Build chronological RBFN DataLoaders.
        """

        if batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1."
            )

        if num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

        samples = (
            self.build_monthly_samples()
        )

        split = (
            self.create_temporal_holdout(
                monthly_samples=samples,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
        )

        train_dataset = (
            RBFNInMemoryDataset(
                split.X_train,
                split.Y_train,
            )
        )

        val_dataset = (
            RBFNInMemoryDataset(
                split.X_val,
                split.Y_val,
            )
        )

        test_dataset = (
            RBFNInMemoryDataset(
                split.X_test,
                split.Y_test,
            )
        )

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

        return (
            train_loader,
            val_loader,
            test_loader,
            split,
        )