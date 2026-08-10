# src/utils/spatial.py
"""
Spatial Helper Utilities.
Provides spatial windowing, coordinate encoding, and spatial weight matrices for raster grids.
"""

from typing import Tuple
import numpy as np


def extract_spatial_windows(
    raster_array: np.ndarray, window_size: int = 3
) -> np.ndarray:
    """Extracts sliding spatial N x N patch windows across a 2D spatial grid."""
    height, width = raster_array.shape[:2]
    half_w = window_size // 2

    padded = np.pad(
        raster_array,
        ((half_w, half_w), (half_w, half_w), (0, 0)),
        mode="reflect",
    )
    patches = []

    for i in range(half_w, height + half_w):
        for j in range(half_w, width + half_w):
            patch = padded[
                i - half_w : i + half_w + 1, j - half_w : j + half_w + 1
            ]
            patches.append(patch.flatten())

    return np.array(patches)


def generate_spatial_coordinates(
    shape: Tuple[int, int], bounds: Tuple[float, float, float, float]
) -> np.ndarray:
    """Generates normalized (x, y) spatial coordinate features for every pixel in a grid."""
    min_x, min_y, max_x, max_y = bounds
    rows, cols = shape

    xs = np.linspace(min_x, max_x, cols)
    ys = np.linspace(min_y, max_y, rows)

    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)

    return coords