# src/models/tuning.py
"""
Grid Search Tuning Engine for RBFN Gaussian Centers and Ridge Regularization Lambda.
"""

from typing import Dict, List
import numpy as np
import torch
from src.models.rbfn import MultiOutputRBFN


class RBFNTuner:
    """Evaluates combinations of K centers and Lambda on temporal holdouts."""

    def __init__(self, config: dict):
        self.config = config

    def grid_search(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        centers_list: List[int] = [30, 50, 100],
        lambda_list: List[float] = [1e-4, 1e-3, 1e-2],
    ) -> Dict[str, float]:
        best_rmse = float("inf")
        best_params = {}

        for k in centers_list:
            for l_reg in lambda_list:
                model = MultiOutputRBFN(
                    in_features=X_train.shape[1],
                    num_centers=k,
                    out_bands=Y_train.shape[1],
                )
                model.fit_centers(X_train)

                A = model._gaussian_rbf(X_train)
                I = torch.eye(A.shape[1])
                W = torch.linalg.inv(A.T @ A + l_reg * I) @ A.T @ Y_train

                model.linear_weights.weight.data = W.T
                model.linear_weights.bias.data.fill_(0.0)

                model.eval()
                with torch.no_grad():
                    preds = model(X_val)
                    rmse = torch.sqrt(torch.mean((preds - Y_val) ** 2)).item()

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {"num_centers": k, "regularization_lambda": l_reg}

        return best_params