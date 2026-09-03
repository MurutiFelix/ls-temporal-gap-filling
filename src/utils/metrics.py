# src/utils/metrics.py

"""
Evaluation metrics for Landsat temporal gap filling.
Provides NaN-safe Pearson correlation, Spearman correlation, RMSE, and MAE.
Supports both per-band Landsat reconstruction evaluation and single-raster
NDVI evaluation against independent reference datasets such as AVHRR and MODIS.
"""

from typing import Dict, Optional, Tuple
import numpy as np


def _validate_matching_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates and converts two arrays to float64 NumPy arrays.
    The arrays must have identical shapes.
    """
    true_array = np.asarray(y_true, dtype=np.float64)
    pred_array = np.asarray(y_pred, dtype=np.float64)

    if true_array.shape != pred_array.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {true_array.shape}, "
            f"but y_pred has shape {pred_array.shape}."
        )

    return true_array, pred_array


def _valid_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns finite paired observations from two matching arrays.
    """
    true_array, pred_array = _validate_matching_arrays(y_true, y_pred)
    mask = np.isfinite(true_array) & np.isfinite(pred_array)

    return true_array[mask], pred_array[mask]


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """
    Computes average ranks with tie handling.
    Ranks are one-based, matching the conventional statistical definition
    used by Spearman correlation.
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Rank calculation expects a one-dimensional array.")

    n = values.size

    if n == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]

    ranks = np.empty(n, dtype=np.float64)

    start = 0

    while start < n:
        end = start + 1

        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1

        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank

        start = end

    return ranks


def compute_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Computes Root Mean Squared Error using finite paired observations.
    Returns:
        RMSE in the same units as the input data.
    """
    true_values, pred_values = _valid_pairs(y_true, y_pred)

    if true_values.size == 0:
        return float("nan")

    return float(
        np.sqrt(np.mean((true_values - pred_values) ** 2))
    )


def compute_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Computes Mean Absolute Error using finite paired observations.
    Returns:
        MAE in the same units as the input data.
    """
    true_values, pred_values = _valid_pairs(y_true, y_pred)

    if true_values.size == 0:
        return float("nan")

    return float(
        np.mean(np.abs(true_values - pred_values))
    )


def compute_pearson_r(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Computes Pearson's correlation coefficient using finite paired values.
    Returns NaN when fewer than two valid observations exist or when either
    variable has zero variance.
    """
    true_values, pred_values = _valid_pairs(y_true, y_pred)

    if true_values.size < 2:
        return float("nan")

    true_centered = true_values - np.mean(true_values)
    pred_centered = pred_values - np.mean(pred_values)

    denominator = np.sqrt(
        np.sum(true_centered ** 2)
        * np.sum(pred_centered ** 2)
    )

    if denominator <= 0.0:
        return float("nan")

    return float(
        np.sum(true_centered * pred_centered) / denominator
    )


def compute_spearman_r(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Computes Spearman's rank correlation coefficient.
    Ties are handled using average ranks. Only finite paired observations
    are included.
    """
    true_values, pred_values = _valid_pairs(y_true, y_pred)

    if true_values.size < 2:
        return float("nan")

    true_ranks = _rankdata_average(true_values)
    pred_ranks = _rankdata_average(pred_values)

    return compute_pearson_r(true_ranks, pred_ranks)


def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes the four core evaluation metrics for two matching arrays.
    This function is intended for scalar spatial products such as NDVI
    comparisons between a gap-filled Landsat raster and an independent
    AVHRR or MODIS reference raster.

    Returns:
        Dictionary containing Pearson r, Spearman r, RMSE, and MAE.
    """
    _validate_matching_arrays(y_true, y_pred)

    return {
        "pearson_r": compute_pearson_r(y_true, y_pred),
        "spearman_r": compute_spearman_r(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
    }


def _to_band_matrix(
    array: np.ndarray,
    spatial_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Converts supported Landsat layouts to (N_pixels, N_bands).
    Supported layouts:
        - (N_pixels, N_bands)
        - (N_bands, Height, Width)

    For a two-dimensional array, spatial_shape is accepted for API
    compatibility but is not required because the input is already assumed
    to be a pixel-by-band matrix.
    """
    array = np.asarray(array, dtype=np.float64)

    if array.ndim == 3:
        n_bands, height, width = array.shape

        if spatial_shape is not None and spatial_shape != (height, width):
            raise ValueError(
                f"Provided spatial_shape {spatial_shape} does not match "
                f"raster shape {(height, width)}."
            )

        return array.reshape(n_bands, height * width).T

    if array.ndim == 2:
        return array

    raise ValueError(
        "Expected a 2D pixel-by-band matrix or a 3D "
        "(bands, height, width) raster cube."
    )


def _validate_band_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spatial_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts and validates two Landsat arrays as pixel-by-band matrices.
    """
    true_matrix = _to_band_matrix(y_true, spatial_shape)
    pred_matrix = _to_band_matrix(y_pred, spatial_shape)

    if true_matrix.shape != pred_matrix.shape:
        raise ValueError(
            f"Shape mismatch after conversion: y_true has shape "
            f"{true_matrix.shape}, but y_pred has shape "
            f"{pred_matrix.shape}."
        )

    return true_matrix, pred_matrix


def evaluate_reconstruction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spatial_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Computes the core evaluation metrics independently for each Landsat band.
    Args:
        y_true:
            Either a matrix of shape (N_pixels, N_bands) or a raster cube
            of shape (N_bands, Height, Width).

        y_pred:
            Predicted array with the same layout and shape as y_true.

        spatial_shape:
            Optional raster shape used to validate a 3D input or retained
            for compatibility with earlier versions of this utility.

    Returns:
        Dictionary containing one metric array per band:
            - pearson_r
            - spearman_r
            - rmse
            - mae

        Each array has shape (N_bands,).
    """
    true_matrix, pred_matrix = _validate_band_matrices(
        y_true,
        y_pred,
        spatial_shape=spatial_shape,
    )

    n_bands = true_matrix.shape[1]

    pearson_scores = np.full(n_bands, np.nan, dtype=np.float32)
    spearman_scores = np.full(n_bands, np.nan, dtype=np.float32)
    rmse_scores = np.full(n_bands, np.nan, dtype=np.float32)
    mae_scores = np.full(n_bands, np.nan, dtype=np.float32)

    for band_index in range(n_bands):
        true_band = true_matrix[:, band_index]
        pred_band = pred_matrix[:, band_index]

        pearson_scores[band_index] = compute_pearson_r(
            true_band,
            pred_band,
        )

        spearman_scores[band_index] = compute_spearman_r(
            true_band,
            pred_band,
        )

        rmse_scores[band_index] = compute_rmse(
            true_band,
            pred_band,
        )

        mae_scores[band_index] = compute_mae(
            true_band,
            pred_band,
        )

    return {
        "pearson_r": pearson_scores,
        "spearman_r": spearman_scores,
        "rmse": rmse_scores,
        "mae": mae_scores,
    }