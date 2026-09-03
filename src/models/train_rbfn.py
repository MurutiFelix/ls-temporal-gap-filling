# src/models/train_rbfn.py

"""
Trainer module for the Radial Basis Function Network (RBFN).
Handles training-only feature and target standardization, fitting K-Means
RBF centres and the Gaussian width parameter, chunked Ridge-regression
statistics, linear readout estimation, chunked inference, evaluation,
and checkpoint saving.

The locked RBFN input contract is:

0:7     Previous Landsat bands
7:14    Next Landsat bands
14      Time distance to previous observation
15      Time distance to next observation
16      Previous observation availability
17      Next observation availability
18      ERA5 precipitation
19      ERA5 temperature
20      DEM

Total input features: 21
Total output features: 7 Landsat bands.

The temporal-availability features at indices 16 and 17 are binary
indicators and are therefore preserved as 0/1 values rather than being
standardized as continuous variables.

All scalers are fitted exclusively on the training partition.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import joblib
import numpy as np
import torch

from src.models.rbfn import RBFN

EXPECTED_IN_FEATURES = 21
EXPECTED_OUT_FEATURES = 7
PREV_AVAILABLE_INDEX = 16
NEXT_AVAILABLE_INDEX = 17
BINARY_FEATURE_INDICES = (
    PREV_AVAILABLE_INDEX,
    NEXT_AVAILABLE_INDEX,
)


class RBFNTrainer:
    """
    Coordinates preprocessing, RBF fitting, Ridge readout estimation,
    inference, evaluation, and checkpoint saving.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        in_features: int,
        out_features: int,
    ) -> None:
        self.config = config

        rbfn_cfg = config.get("model", {}).get("rbfn", {})

        self.num_centers = int(
            rbfn_cfg.get("num_centers", 50)
        )
        self.regularization_lambda = float(
            rbfn_cfg.get("regularization_lambda", 1e-2)
        )
        self.gamma_multiplier = float(
            rbfn_cfg.get("gamma_multiplier", 1.0)
        )
        self.random_state = int(
            config.get("project", {}).get("seed", 42)
        )
        self.chunk_size = int(
            rbfn_cfg.get("chunk_size", 8192)
        )

        if in_features != EXPECTED_IN_FEATURES:
            raise ValueError(
                "RBFNTrainer requires exactly "
                f"{EXPECTED_IN_FEATURES} input features; "
                f"received {in_features}."
            )

        if out_features != EXPECTED_OUT_FEATURES:
            raise ValueError(
                "RBFNTrainer requires exactly "
                f"{EXPECTED_OUT_FEATURES} output features; "
                f"received {out_features}."
            )

        if self.num_centers <= 0:
            raise ValueError(
                "num_centers must be greater than zero."
            )

        if self.regularization_lambda < 0.0:
            raise ValueError(
                "regularization_lambda cannot be negative."
            )

        if self.gamma_multiplier <= 0.0:
            raise ValueError(
                "gamma_multiplier must be greater than zero."
            )

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = RBFN(
            in_features=in_features,
            num_centers=self.num_centers,
            out_features=out_features,
            rbf_chunk_size=self.chunk_size,
        ).to(self.device)

        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.y_mean: Optional[np.ndarray] = None
        self.y_std: Optional[np.ndarray] = None

        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the trainer has completed model fitting."""
        return self._fitted

    @staticmethod
    def _validate_2d_array(
        array: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """Validate and convert an input matrix to contiguous float32."""
        array = np.asarray(array, dtype=np.float32)

        if array.ndim != 2:
            raise ValueError(
                f"{name} must be a 2-dimensional array."
            )

        if array.shape[0] == 0:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} contains non-finite values."
            )

        return np.ascontiguousarray(
            array,
            dtype=np.float32,
        )

    @staticmethod
    def _validate_binary_features(
        X: np.ndarray,
        context: str,
    ) -> None:
        """Validate the temporal-availability features."""
        for index in BINARY_FEATURE_INDICES:
            values = X[:, index]

            if not np.all(
                np.logical_or(
                    np.isclose(values, 0.0),
                    np.isclose(values, 1.0),
                )
            ):
                raise ValueError(
                    f"{context} feature index {index} must contain "
                    "only binary 0/1 availability values."
                )

    def _validate_feature_target_pair(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        context: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate a feature-target matrix pair."""
        X = self._validate_2d_array(
            X,
            f"{context} features",
        )
        Y = self._validate_2d_array(
            Y,
            f"{context} targets",
        )

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"{context} features and targets must contain "
                "the same number of samples."
            )

        if X.shape[1] != self.model.in_features:
            raise ValueError(
                f"{context} feature dimension mismatch: "
                f"expected {self.model.in_features}, "
                f"received {X.shape[1]}."
            )

        if Y.shape[1] != self.model.out_features:
            raise ValueError(
                f"{context} target dimension mismatch: "
                f"expected {self.model.out_features}, "
                f"received {Y.shape[1]}."
            )

        self._validate_binary_features(
            X,
            context,
        )

        return X, Y

    def fit_scalers(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit feature and target standardization using training data only.

        Continuous input features are standardized. The two temporal
        availability indicators remain unchanged as binary 0/1 features.
        """
        X_train, Y_train = self._validate_feature_target_pair(
            X_train,
            Y_train,
            "Training",
        )

        self.x_mean = np.mean(
            X_train,
            axis=0,
            keepdims=True,
            dtype=np.float64,
        ).astype(np.float32)

        self.x_std = np.std(
            X_train,
            axis=0,
            keepdims=True,
            dtype=np.float64,
        ).astype(np.float32)

        self.x_std[self.x_std < 1e-8] = 1.0

        for index in BINARY_FEATURE_INDICES:
            self.x_mean[:, index] = 0.0
            self.x_std[:, index] = 1.0

        self.y_mean = np.mean(
            Y_train,
            axis=0,
            keepdims=True,
            dtype=np.float64,
        ).astype(np.float32)

        self.y_std = np.std(
            Y_train,
            axis=0,
            keepdims=True,
            dtype=np.float64,
        ).astype(np.float32)

        self.y_std[self.y_std < 1e-8] = 1.0

        X_train_scaled = (
            X_train - self.x_mean
        ) / self.x_std

        Y_train_scaled = (
            Y_train - self.y_mean
        ) / self.y_std

        self._validate_binary_features(
            X_train_scaled,
            "Scaled training",
        )

        return (
            np.ascontiguousarray(
                X_train_scaled,
                dtype=np.float32,
            ),
            np.ascontiguousarray(
                Y_train_scaled,
                dtype=np.float32,
            ),
        )

    def transform_features(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Transform features using the training-fitted feature scaler."""
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError(
                "Feature scalers have not been fitted."
            )

        X = self._validate_2d_array(
            X,
            "Feature matrix",
        )

        if X.shape[1] != self.model.in_features:
            raise ValueError(
                f"Expected {self.model.in_features} features; "
                f"received {X.shape[1]}."
            )

        self._validate_binary_features(
            X,
            "Feature matrix",
        )

        X_scaled = (
            X - self.x_mean
        ) / self.x_std

        self._validate_binary_features(
            X_scaled,
            "Scaled feature matrix",
        )

        return np.ascontiguousarray(
            X_scaled,
            dtype=np.float32,
        )

    def transform_targets(
        self,
        Y: np.ndarray,
    ) -> np.ndarray:
        """Transform targets using the training-fitted target scaler."""
        if self.y_mean is None or self.y_std is None:
            raise RuntimeError(
                "Target scalers have not been fitted."
            )

        Y = self._validate_2d_array(
            Y,
            "Target matrix",
        )

        if Y.shape[1] != self.model.out_features:
            raise ValueError(
                f"Expected {self.model.out_features} targets; "
                f"received {Y.shape[1]}."
            )

        Y_scaled = (
            Y - self.y_mean
        ) / self.y_std

        return np.ascontiguousarray(
            Y_scaled,
            dtype=np.float32,
        )

    def inverse_transform_targets(
        self,
        Y_scaled: np.ndarray,
    ) -> np.ndarray:
        """Convert standardized targets back to physical units."""
        if self.y_mean is None or self.y_std is None:
            raise RuntimeError(
                "Target scalers have not been fitted."
            )

        Y_scaled = self._validate_2d_array(
            Y_scaled,
            "Scaled target matrix",
        )

        if Y_scaled.shape[1] != self.model.out_features:
            raise ValueError(
                f"Expected {self.model.out_features} targets; "
                f"received {Y_scaled.shape[1]}."
            )

        Y = (
            Y_scaled * self.y_std
        ) + self.y_mean

        return np.ascontiguousarray(
            Y,
            dtype=np.float32,
        )

    def _iter_chunks(
        self,
        n_samples: int,
    ) -> Iterator[Tuple[int, int]]:
        """Yield start and end indices for chunked processing."""
        for start in range(
            0,
            n_samples,
            self.chunk_size,
        ):
            end = min(
                start + self.chunk_size,
                n_samples,
            )
            yield start, end

    def _accumulate_ridge_statistics(
        self,
        X_train_scaled: np.ndarray,
        Y_train_scaled: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Accumulate Ridge-regression sufficient statistics in chunks.

        Returns:
            Phi^T Phi
            Phi^T Y
            sum(Phi)
            sum(Y)
        """
        X_train_scaled, Y_train_scaled = (
            self._validate_feature_target_pair(
                X_train_scaled,
                Y_train_scaled,
                "Scaled training",
            )
        )

        n_centers = self.model.num_centers
        n_outputs = self.model.out_features

        phi_t_phi = np.zeros(
            (n_centers, n_centers),
            dtype=np.float64,
        )
        phi_t_y = np.zeros(
            (n_centers, n_outputs),
            dtype=np.float64,
        )
        phi_sum = np.zeros(
            n_centers,
            dtype=np.float64,
        )
        y_sum = np.zeros(
            n_outputs,
            dtype=np.float64,
        )

        self.model.eval()

        with torch.no_grad():
            for start, end in self._iter_chunks(
                X_train_scaled.shape[0]
            ):
                X_chunk = torch.as_tensor(
                    X_train_scaled[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )

                Y_chunk = Y_train_scaled[start:end]

                phi_chunk = (
                    self.model
                    .get_rbf_features(X_chunk)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float64,
                        copy=False,
                    )
                )

                phi_t_phi += (
                    phi_chunk.T @ phi_chunk
                )

                phi_t_y += (
                    phi_chunk.T @ Y_chunk
                )

                phi_sum += np.sum(
                    phi_chunk,
                    axis=0,
                    dtype=np.float64,
                )

                y_sum += np.sum(
                    Y_chunk,
                    axis=0,
                    dtype=np.float64,
                )

        return (
            phi_t_phi,
            phi_t_y,
            phi_sum,
            y_sum,
        )

    def fit_ridge(
        self,
        X_train_scaled: np.ndarray,
        Y_train_scaled: np.ndarray,
        user_gamma: Optional[float] = None,
    ) -> float:
        """
        Fit RBF centres, Gaussian gamma, and Ridge readout parameters.

        Returns:
            Training mean squared error in standardized target space.
        """
        X_train_scaled, Y_train_scaled = (
            self._validate_feature_target_pair(
                X_train_scaled,
                Y_train_scaled,
                "Scaled training",
            )
        )

        if X_train_scaled.shape[0] < self.num_centers:
            raise ValueError(
                "The number of training samples must be at least "
                "the number of RBF centres."
            )

        self.model.fit_centers_and_gamma(
            X=X_train_scaled,
            gamma=user_gamma,
            random_state=self.random_state,
            gamma_multiplier=self.gamma_multiplier,
        )

        self.model.to(self.device)

        (
            phi_t_phi,
            phi_t_y,
            phi_sum,
            y_sum,
        ) = self._accumulate_ridge_statistics(
            X_train_scaled,
            Y_train_scaled,
        )

        n_samples = float(
            X_train_scaled.shape[0]
        )
        n_centers = self.model.num_centers
        n_outputs = self.model.out_features

        augmented_size = n_centers + 1

        lhs = np.zeros(
            (augmented_size, augmented_size),
            dtype=np.float64,
        )

        rhs = np.zeros(
            (augmented_size, n_outputs),
            dtype=np.float64,
        )

        regularization = (
            self.regularization_lambda
            * np.eye(
                n_centers,
                dtype=np.float64,
            )
        )

        lhs[
            :n_centers,
            :n_centers,
        ] = phi_t_phi + regularization

        lhs[
            :n_centers,
            n_centers,
        ] = phi_sum

        lhs[
            n_centers,
            :n_centers,
        ] = phi_sum

        lhs[
            n_centers,
            n_centers,
        ] = n_samples

        rhs[
            :n_centers,
            :,
        ] = phi_t_y

        rhs[
            n_centers,
            :,
        ] = y_sum

        try:
            solution = np.linalg.solve(
                lhs,
                rhs,
            )
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(
                lhs,
                rhs,
                rcond=None,
            )[0]

        coefficients = solution[
            :n_centers,
            :
        ]

        intercept = solution[
            n_centers,
            :
        ]

        self.model.set_readout_parameters(
            coefficients=np.ascontiguousarray(
                coefficients.T,
                dtype=np.float32,
            ),
            intercept=np.ascontiguousarray(
                intercept,
                dtype=np.float32,
            ),
        )

        self._fitted = True

        return self._calculate_scaled_mse(
            X_train_scaled,
            Y_train_scaled,
        )

    def _predict_scaled(
        self,
        X_scaled: np.ndarray,
    ) -> np.ndarray:
        """Generate standardized predictions in chunks."""
        if not self._fitted:
            raise RuntimeError(
                "The RBFN trainer has not been fitted."
            )

        X_scaled = self._validate_2d_array(
            X_scaled,
            "Scaled feature matrix",
        )

        if X_scaled.shape[1] != self.model.in_features:
            raise ValueError(
                f"Expected {self.model.in_features} features; "
                f"received {X_scaled.shape[1]}."
            )

        self._validate_binary_features(
            X_scaled,
            "Scaled feature matrix",
        )

        predictions = np.empty(
            (
                X_scaled.shape[0],
                self.model.out_features,
            ),
            dtype=np.float32,
        )

        self.model.eval()

        with torch.no_grad():
            for start, end in self._iter_chunks(
                X_scaled.shape[0]
            ):
                X_chunk = torch.as_tensor(
                    X_scaled[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )

                predictions[start:end] = (
                    self.model(X_chunk)
                    .detach()
                    .cpu()
                    .numpy()
                )

        return predictions

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Predict physical Landsat band values."""
        X_scaled = self.transform_features(X)

        Y_pred_scaled = self._predict_scaled(
            X_scaled
        )

        return self.inverse_transform_targets(
            Y_pred_scaled
        )

    def _calculate_scaled_mse(
        self,
        X_scaled: np.ndarray,
        Y_scaled: np.ndarray,
    ) -> float:
        """Calculate mean squared error in standardized target space."""
        X_scaled, Y_scaled = (
            self._validate_feature_target_pair(
                X_scaled,
                Y_scaled,
                "Scaled evaluation",
            )
        )

        squared_error_sum = 0.0
        n_values = float(
            Y_scaled.size
        )

        self.model.eval()

        with torch.no_grad():
            for start, end in self._iter_chunks(
                X_scaled.shape[0]
            ):
                X_chunk = torch.as_tensor(
                    X_scaled[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )

                prediction_chunk = (
                    self.model(X_chunk)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float64,
                        copy=False,
                    )
                )

                target_chunk = (
                    Y_scaled[start:end]
                    .astype(
                        np.float64,
                        copy=False,
                    )
                )

                squared_error_sum += float(
                    np.sum(
                        (
                            target_chunk
                            - prediction_chunk
                        ) ** 2
                    )
                )

        return squared_error_sum / n_values

    def evaluate(
        self,
        X_eval_scaled: np.ndarray,
        Y_eval_scaled: np.ndarray,
        Y_eval_raw: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate the fitted model on a dataset partition."""
        if not self._fitted:
            raise RuntimeError(
                "The RBFN trainer has not been fitted."
            )

        X_eval_scaled, Y_eval_scaled = (
            self._validate_feature_target_pair(
                X_eval_scaled,
                Y_eval_scaled,
                "Evaluation",
            )
        )

        Y_eval_raw = self._validate_2d_array(
            Y_eval_raw,
            "Physical evaluation targets",
        )

        if Y_eval_raw.shape != Y_eval_scaled.shape:
            raise ValueError(
                "Scaled and physical evaluation targets must "
                "have identical shapes."
            )

        Y_pred_scaled = self._predict_scaled(
            X_eval_scaled
        )

        Y_pred_raw = self.inverse_transform_targets(
            Y_pred_scaled
        )

        mse_scaled = float(
            np.mean(
                (
                    Y_eval_scaled
                    - Y_pred_scaled
                ) ** 2
            )
        )

        rmse_physical = float(
            np.sqrt(
                np.mean(
                    (
                        Y_eval_raw
                        - Y_pred_raw
                    ) ** 2
                )
            )
        )

        ss_res = float(
            np.sum(
                (
                    Y_eval_raw
                    - Y_pred_raw
                ) ** 2
            )
        )

        target_mean = np.mean(
            Y_eval_raw,
            axis=0,
            keepdims=True,
        )

        ss_tot = float(
            np.sum(
                (
                    Y_eval_raw
                    - target_mean
                ) ** 2
            )
        )

        r2_score = (
            float("nan")
            if ss_tot <= 1e-12
            else float(
                1.0
                - (
                    ss_res
                    / ss_tot
                )
            )
        )

        return {
            "eval_mse_scaled": mse_scaled,
            "eval_rmse_physical": rmse_physical,
            "eval_r2": r2_score,
        }

    def save_checkpoint(
        self,
        model_path: str,
        scaler_path: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Save the fitted RBFN state and training-fitted scalers."""
        if not self._fitted:
            raise RuntimeError(
                "Cannot save an RBFN checkpoint before fitting."
            )

        if (
            self.x_mean is None
            or self.x_std is None
            or self.y_mean is None
            or self.y_std is None
        ):
            raise RuntimeError(
                "Cannot save checkpoint before fitting scalers."
            )

        model_path_obj = Path(model_path)
        scaler_path_obj = Path(scaler_path)

        model_path_obj.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        scaler_path_obj.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        state_dict_cpu = {
            key: value.detach().cpu()
            for key, value in self.model.state_dict().items()
        }

        torch.save(
            {
                "state_dict": state_dict_cpu,
                "config": self.config,
                "metadata": metadata,
                "in_features": self.model.in_features,
                "num_centers": self.model.num_centers,
                "out_features": self.model.out_features,
            },
            model_path_obj,
        )

        joblib.dump(
            {
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "binary_feature_indices": list(
                    BINARY_FEATURE_INDICES
                ),
            },
            scaler_path_obj,
        )