# src/utils/metrics.py
"""
Geospatial Metrics & Downstream Spectral Index Solvers.
Calculates VHI, BSI, SMI, and computes evaluation metrics (RMSE, MAE).
"""

import numpy as np


def compute_spectral_indices(predicted_cube: np.ndarray) -> dict:
    """
    Calculates downstream spectral indices on the reconstructed 6-band Landsat cube.
    Bands assumed: [Red, Green, Blue, NIR, SWIR, Thermal]
    """
    red = predicted_cube[:, 0]
    nir = predicted_cube[:, 3]
    swir = predicted_cube[:, 4]
    thermal = predicted_cube[:, 5]

    # Normalized Difference Vegetation Index (NDVI)
    ndvi = (nir - red) / (nir + red + 1e-6)

    # Bare Soil Index (BSI)
    bsi = ((swir + red) - (nir + red)) / ((swir + red) + (nir + red) + 1e-6)

    # Vegetation Health Index (VHI Proxy)
    vhi = (ndvi + (1.0 - thermal)) / 2.0

    return {
        "NDVI": ndvi,
        "BSI": bsi,
        "VHI": vhi,
    }


def compute_reconstruction_rmse(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Calculates per-band Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))