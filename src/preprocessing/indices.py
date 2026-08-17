# src/preprocessing/indices.py
"""
Calculates spectral indices (NDVI, BSI, VHI, SMI) on 7-band Landsat output matrices/cubes.
Assumed Landsat Bands: [Red, Green, Blue, NIR, SWIR1, SWIR2, Thermal]
"""

import numpy as np


def compute_spectral_indices(predicted_cube: np.ndarray) -> dict:
    """
    Computes spectral indices from reconstructed 7-band Landsat array.
    Shape: (N_samples, 7) or (Height, Width, 7)
    """
    red = predicted_cube[..., 0]
    blue = predicted_cube[..., 2]
    nir = predicted_cube[..., 3]
    swir1 = predicted_cube[..., 4]
    thermal = predicted_cube[..., 6]

    # Normalized Difference Vegetation Index
    ndvi = (nir - red) / (nir + red + 1e-6)

    # Bare Soil Index — matches finalized formula: [(SWIR1+R)-(NIR+B)] / [(SWIR1+R)+(NIR+B)]
    bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-6)

    # Vegetation Health Index Proxy
    vhi = (ndvi + (1.0 - thermal)) / 2.0

    return {
        "NDVI": ndvi,
        "BSI": bsi,
        "VHI": vhi,
    }