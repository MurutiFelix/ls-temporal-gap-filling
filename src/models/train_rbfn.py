# src/models/train_rbfn.py

"""
Trainer module for the Radial Basis Function Network (RBFN).

Handles feature and target standardization, fitting K-Means centres and gamma,
estimating linear readout weights via Ridge regression, and evaluating models
on physical scales.
"""

from typing import Any, Dict, Tuple
import joblib
import numpy as np
import torch

from src.models.rbfn import RBFNetwork


class RBFNTrainer:
    """
    Coordinates training, transformation, and evaluation for RBFNetwork.
    """

    def __init__(
        self,
        config: dict,
        in_features: int,
        out_features: int,
    ):
        self.config = config
        rbfn_cfg = config.get("model", {}).get("rbfn", {})

        self.num_centers = rbfn_cfg.get("num_centers", 50)
        self.regularization_lambda = rbfn_cfg.get("regularization_lambda", 1e-2)
        self.gamma_multiplier = rbfn_cfg.get("gamma_multiplier", 1.0)
        self.random_state = config.get("project", {}).get("seed", 42)

        self.model = RBFNetwork(
            in_features=in_features,
            num_centers=self.num_centers,
            out_features=out_features,
        )

        # Standardization parameters
        self.x_mean: np.ndarray = None
        self.x_std: np.ndarray = None
        self.y_mean: np.ndarray = None
        self.y_std: np.ndarray = None

    def fit_scalers(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit standardization parameters on training data and return scaled matrices.
        """
        self.x_mean = np.mean(X_train, axis=0, keepdims=True)
        self.x_std = np.std(X_train, axis=0, keepdims=True)
        self.x_std[self.x_std < 1e-8] = 1.0

        self.y_mean = np.mean(Y_train, axis=0, keepdims=True)
        self.y_std = np.std(Y_train, axis=0, keepdims=True)
        self.y_std[self.y_std < 1e-8] = 1.0

        X_train_scaled = (X_train - self.x_mean) / self.x_std
        Y_train_scaled = (Y_train - self.y_mean) / self.y_std

        return X_train_scaled, Y_train_scaled

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Standardize feature matrix using fitted parameters."""
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("Scalers have not been fitted yet.")
        return (X - self.x_mean) / self.x_std

    def transform_targets(self, Y: np.ndarray) -> np.ndarray:
        """Standardize target matrix using fitted parameters."""
        if self.y_mean is None or self.y_std is None:
            raise RuntimeError("Scalers have not been fitted yet.")
        return (Y - self.y_mean) / self.y_std

    def fit_ridge(
        self,
        X_train_scaled: np.ndarray,
        Y_train_scaled: np.ndarray,
    ) -> float:
        """
        Fit K-Means centres, compute gamma, and solve linear readout weights using
        Ridge regression in closed form:

            W = (Phi^T * Phi + lambda * I)^(-1) * Phi^T * Y
        """
        # 1. Fit centers and gamma on training features
        self.model.fit_centers_and_gamma(
            X_train_scaled=X_train_scaled,
            random_state=self.random_state,
            gamma_multiplier=self.gamma_multiplier,
        )

        # 2. Compute activation matrix Phi
        X_tensor = torch.from_numpy(X_train_scaled).float()
        with torch.no_grad():
            phi = self.model.radial_basis_function(X_tensor).numpy()

        # 3. Closed-form Ridge solution
        n_samples, n_centers = phi.shape
        identity = np.eye(n_centers, dtype=np.float32)

        lhs = (phi.T @ phi) + (self.regularization_lambda * identity)
        rhs = phi.T @ Y_train_scaled

        weights = np.linalg.solve(lhs, rhs)  # Shape: (K, M)

        # 4. Transfer weights to linear layer
        self.model.linear.weight.data = torch.from_numpy(weights.T).float()
        self.model.linear.bias.data.zero_()

        # 5. Return training MSE on scaled data
        with torch.no_grad():
            Y_pred_scaled = self.model(X_tensor).numpy()

        return float(np.mean((Y_train_scaled - Y_pred_scaled) ** 2))

    def evaluate(
        self,
        X_eval_scaled: np.ndarray,
        Y_eval_scaled: np.ndarray,
        Y_eval_raw: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model performance on scaled and physical reflectance scales.
        """
        X_tensor = torch.from_numpy(X_eval_scaled).float()

        with torch.no_grad():
            Y_pred_scaled = self.model(X_tensor).numpy()

        # Unscale predictions back to physical scale
        Y_pred_raw = (Y_pred_scaled * self.y_std) + self.y_mean

        # Metrics
        mse_scaled = float(np.mean((Y_eval_scaled - Y_pred_scaled) ** 2))
        rmse_physical = float(np.sqrt(np.mean((Y_eval_raw - Y_pred_raw) ** 2)))

        ss_res = np.sum((Y_eval_raw - Y_pred_raw) ** 2)
        ss_tot = np.sum((Y_eval_raw - np.mean(Y_eval_raw, axis=0)) ** 2)
        r2_score = float(1.0 - (ss_res / (ss_tot + 1e-8)))

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
        """Save PyTorch weights and standardization parameters."""
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config,
                "metadata": metadata,
            },
            model_path,
        )

        joblib.dump(
            {
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
            },
            scaler_path,
        )