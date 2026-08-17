# src/models/train_rbfn.py
"""
Deep Learning Training Modules and Loss Tracking for PyTorch Models.
Encapsulates iterative training, validation passes, and model checkpointing.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class RBFNTrainer:
    """Manages training loops, closed-form/gradient updates, and checkpointing for RBFN."""

    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config

    def fit_ridge(
        self, X_train: torch.Tensor, Y_train: torch.Tensor, lambda_reg: float = 1e-3
    ):
        """Solves RBFN output weights using closed-form regularized Ridge regression."""
        self.model.fit_centers(X_train)
        A = self.model._gaussian_rbf(X_train)

        I = torch.eye(A.shape[1], device=X_train.device)
        W = torch.linalg.inv(A.T @ A + lambda_reg * I) @ A.T @ Y_train

        self.model.linear_weights.weight.data = W.T
        self.model.linear_weights.bias.data.fill_(0.0)
        print("  ✓ RBFN output weights successfully computed via Ridge solution.")

    def evaluate(self, X_val: torch.Tensor, Y_val: torch.Tensor) -> float:
        """Evaluates validation Loss/RMSE."""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_val)
            rmse = torch.sqrt(torch.mean((preds - Y_val) ** 2)).item()
        return rmse