# src/preprocessing/dataset.py
"""
PyTorch Dataset/DataLoader for RBFN streaming, plus temporal train/val splitting.
"""

from typing import Tuple
import numpy as np


class GapFillDataset:
    """Handles temporal splitting and will house the PyTorch Dataset/DataLoader for RBFN streaming."""

    def __init__(self, config: dict):
        self.config = config

    def create_temporal_holdout(
        self, X: np.ndarray, Y: np.ndarray, val_ratio: float = 0.22
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits time series matrices into training and validation sets (e.g., 18/5 holdout)."""
        n_samples = X.shape[0]
        split_idx = int(n_samples * (1.0 - val_ratio))

        X_train, X_val = X[:split_idx], X[split_idx:]
        Y_train, Y_val = Y[:split_idx], Y[split_idx:]

        return X_train, X_val, Y_train, Y_val

    # TODO: add PyTorch Dataset/__getitem__/__len__ + DataLoader construction here