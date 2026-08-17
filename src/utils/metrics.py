# src/utils/metrics.py
"""
Evaluates spatial gap reconstruction performance (RMSE, MAE, SSIM).
"""

import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates per-band Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates per-band Mean Absolute Error (MAE)."""
    return np.mean(np.abs(y_true - y_pred), axis=0)


def compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculates Coefficient of Determination (R²) per band."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    return 1.0 - (ss_res / (ss_tot + 1e-8))