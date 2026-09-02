# src/utils/spatial.py

"""
Spatial utilities for Landsat temporal gap filling.

Provides reusable spatial distance and weighting operations for the
RBFN pipeline. Target-grid construction and pixel-coordinate generation
remain under RasterProcessor and DatasetBuilder respectively.
"""

from typing import Optional, Tuple

import numpy as np


def validate_spatial_shape(
    array: np.ndarray,
    expected_shape: Tuple[int, int],
) -> None:
    """Validate that a raster or raster cube matches the target grid."""
    if array.ndim < 2:
        raise ValueError(
            f"Expected an array with at least two spatial dimensions, "
            f"received shape {array.shape}."
        )

    actual_shape = array.shape[:2]

    if actual_shape != expected_shape:
        raise ValueError(
            f"Spatial shape mismatch: expected {expected_shape}, "
            f"received {actual_shape}."
        )


def spatial_distance(
    coordinates: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean spatial distance from each coordinate to a reference.

    Args:
        coordinates:
            Array of shape (N, 2).

        reference:
            Coordinate of shape (2,).

    Returns:
        One-dimensional array of distances with shape (N,).
    """
    coordinates = np.asarray(
        coordinates,
        dtype=np.float32,
    )

    reference = np.asarray(
        reference,
        dtype=np.float32,
    )

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "coordinates must have shape (N, 2)."
        )

    if reference.shape != (2,):
        raise ValueError(
            "reference must have shape (2,)."
        )

    differences = coordinates - reference

    return np.sqrt(
        np.sum(
            differences ** 2,
            axis=1,
        )
    )


def gaussian_spatial_weights(
    distances: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """
    Compute Gaussian spatial weights from distances.

    The weighting function is:

        w(d) = exp(-d² / (2σ²))

    where σ is the spatial bandwidth.
    """
    if bandwidth <= 0:
        raise ValueError(
            "bandwidth must be greater than zero."
        )

    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    return np.exp(
        -(
            distances ** 2
        )
        / (
            2.0 * bandwidth ** 2
        )
    ).astype(
        np.float32,
        copy=False,
    )


def normalize_coordinates(
    coordinates: np.ndarray,
    minimum: Optional[np.ndarray] = None,
    maximum: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Normalize coordinate features independently to approximately [-1, 1].
    """
    coordinates = np.asarray(
        coordinates,
        dtype=np.float32,
    )

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "coordinates must have shape (N, 2)."
        )

    if minimum is None:
        minimum = np.min(
            coordinates,
            axis=0,
        )

    if maximum is None:
        maximum = np.max(
            coordinates,
            axis=0,
        )

    minimum = np.asarray(
        minimum,
        dtype=np.float32,
    )

    maximum = np.asarray(
        maximum,
        dtype=np.float32,
    )

    span = maximum - minimum

    if np.any(span <= 0):
        raise ValueError(
            "Coordinate ranges must be greater than zero."
        )

    normalized = (
        2.0
        * (
            coordinates - minimum
        )
        / span
        - 1.0
    )

    return normalized.astype(
        np.float32,
        copy=False,
    )