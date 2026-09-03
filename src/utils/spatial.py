# src/utils/spatial.py

"""
Spatial utilities for Landsat temporal gap filling.
Provides reusable validation and coordinate-distance operations used by
the preprocessing and evaluation pipeline. Target-grid construction,
raster alignment, reprojection, and coordinate generation remain under
RasterProcessor and DatasetBuilder.
"""

from typing import Tuple
import numpy as np


def validate_raster_shape(
    array: np.ndarray,
    expected_shape: Tuple[int, int],
) -> None:
    """
    Validate the spatial dimensions of a raster or raster cube.
    Supported layouts are:

        - (Height, Width)
        - (Bands, Height, Width)

    Args:
        array:
            Raster or raster-cube array.

        expected_shape:
            Expected spatial shape as (Height, Width).
    """
    array = np.asarray(array)

    if array.ndim == 2:
        actual_shape = array.shape

    elif array.ndim == 3:
        actual_shape = array.shape[-2:]

    else:
        raise ValueError(
            "Expected a 2D raster or 3D raster cube, "
            f"received shape {array.shape}."
        )

    if tuple(actual_shape) != tuple(expected_shape):
        raise ValueError(
            f"Spatial shape mismatch: expected {expected_shape}, "
            f"received {actual_shape}."
        )


def spatial_distance(
    coordinates: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Compute Euclidean distance from each coordinate to a reference point.
    Args:
        coordinates:
            Array with shape (N, 2).

        reference:
            Coordinate with shape (2,).

    Returns:
        One-dimensional array of distances with shape (N,).
    """
    coordinates = np.asarray(coordinates, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "coordinates must have shape (N, 2)."
        )

    if reference.shape != (2,):
        raise ValueError(
            "reference must have shape (2,)."
        )

    if not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "coordinates must contain only finite values."
        )

    if not np.all(np.isfinite(reference)):
        raise ValueError(
            "reference must contain only finite values."
        )

    differences = coordinates - reference

    return np.sqrt(
        np.sum(
            differences ** 2,
            axis=1,
        )
    ).astype(
        np.float32,
        copy=False,
    )


def normalize_coordinates(
    coordinates: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> np.ndarray:
    """
    Normalize two-dimensional coordinates independently to [-1, 1].
    Args:
        coordinates:
            Array with shape (N, 2).

        minimum:
            Minimum coordinate values with shape (2,).

        maximum:
            Maximum coordinate values with shape (2,).

    Returns:
        Normalized coordinates with shape (N, 2).
    """
    coordinates = np.asarray(coordinates, dtype=np.float32)
    minimum = np.asarray(minimum, dtype=np.float32)
    maximum = np.asarray(maximum, dtype=np.float32)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            "coordinates must have shape (N, 2)."
        )

    if minimum.shape != (2,):
        raise ValueError(
            "minimum must have shape (2,)."
        )

    if maximum.shape != (2,):
        raise ValueError(
            "maximum must have shape (2,)."
        )

    if not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "coordinates must contain only finite values."
        )

    if not np.all(np.isfinite(minimum)):
        raise ValueError(
            "minimum must contain only finite values."
        )

    if not np.all(np.isfinite(maximum)):
        raise ValueError(
            "maximum must contain only finite values."
        )

    span = maximum - minimum

    if np.any(span <= 0.0):
        raise ValueError(
            "Coordinate ranges must be greater than zero."
        )

    normalized = (
        2.0 * (coordinates - minimum) / span
    ) - 1.0

    return normalized.astype(
        np.float32,
        copy=False,
    )