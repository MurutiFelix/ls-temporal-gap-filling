# src/models/tuning.py
"""
Hyperparameter Tuning Engine for the RBFN.
Uses a held-out validation set to select num_centers and regularization_lambda
via actual RBFN train+eval per combination (not a placeholder score).
"""

from typing import Dict
import torch

from src.models.rbfn import MultiOutputRBFN
from src.models.train_rbfn import RBFNTrainer


class HyperparameterTuner:
    """Tunes RBFN parameters (num_centers, lambda) via grid search on real validation RMSE."""

    def __init__(self, config: dict):
        self.config = config

    def grid_search_rbfn(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        center_candidates: list = [30, 50, 100],
        lambda_candidates: list = [1e-4, 1e-3, 1e-2],
    ) -> Dict[str, float]:
        """Evaluates hyperparameter combinations to minimize real validation RMSE."""
        best_rmse = float("inf")
        best_params = {}

        for k in center_candidates:
            for reg in lambda_candidates:
                model = MultiOutputRBFN(
                    in_features=X_train.shape[1],
                    num_centers=k,
                    out_bands=Y_train.shape[1],
                )
                trainer = RBFNTrainer(model, self.config)
                trainer.fit_ridge(X_train, Y_train, lambda_reg=reg)
                score = trainer.evaluate(X_val, Y_val)

                if score < best_rmse:
                    best_rmse = score
                    best_params = {"num_centers": k, "regularization_lambda": reg}

        print(f"  ✓ Optimal Hyperparameters Found: {best_params} (val RMSE={best_rmse:.5f})")
        return best_params