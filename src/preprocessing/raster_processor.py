# src/preprocessing/raster_processor.py
"""
Geospatial Raster Processor for Landsat, MODIS/AVHRR, and ERA5 Modalities.
Handles raster alignment, scale standardization (0-1), and 3D-to-2D matrix flattening.
"""

from pathlib import Path
import numpy as np


class RasterProcessor:
    """Handles raster transformations, normalization, and feature array concatenation."""

    def __init__(self, config: dict):
        self.config = config
        self.num_bands = config["landsat"]["num_bands"]

    def normalize_reflectance(self, arr: np.ndarray) -> np.ndarray:
        """Scales optical reflectance strictly between 0.0 and 1.0."""
        return np.clip(arr, 0.0, 1.0)

    def assemble_feature_matrix(
        self,
        coarse_bands: np.ndarray,
        climate_data: np.ndarray,
        static_norms: np.ndarray,
    ) -> np.ndarray:
        """
        Flattens spatial dimensions and concatenates predictor arrays into matrix X.
        Shape: (N_samples, N_features)
        """
        n_samples = coarse_bands.shape[0]

        f_coarse = coarse_bands.reshape(n_samples, -1)
        f_climate = climate_data.reshape(n_samples, -1)
        f_norms = static_norms.reshape(n_samples, -1)

        return np.hstack([f_coarse, f_climate, f_norms])