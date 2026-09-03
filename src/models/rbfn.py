# src/models/rbfn.py

"""
Radial Basis Function Network (RBFN) PyTorch Model.

Implements a non-linear Gaussian RBF hidden layer using K-Means-derived
centres, followed by a linear readout layer for multi-output Landsat
reflectance prediction.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class RBFN(nn.Module):
    """
    Multi-output Gaussian Radial Basis Function Network.
    """

    PREVIOUS_BAND_START = 0
    PREVIOUS_BAND_END = 7

    NEXT_BAND_START = 7
    NEXT_BAND_END = 14

    DT_PREV_INDEX = 14
    DT_NEXT_INDEX = 15

    PREV_AVAILABLE_INDEX = 16
    NEXT_AVAILABLE_INDEX = 17

    ERA5_PRECIP_INDEX = 18
    ERA5_TEMP_INDEX = 19
    DEM_INDEX = 20

    EXPECTED_IN_FEATURES = 21
    EXPECTED_OUT_FEATURES = 7

    def __init__(
        self,
        in_features: int = EXPECTED_IN_FEATURES,
        num_centers: int = 500,
        out_features: int = EXPECTED_OUT_FEATURES,
        rbf_chunk_size: int = 8192,
    ) -> None:
        super().__init__()

        if in_features != self.EXPECTED_IN_FEATURES:
            raise ValueError(
                f"RBFN requires {self.EXPECTED_IN_FEATURES} input features, "
                f"but received {in_features}."
            )

        if out_features != self.EXPECTED_OUT_FEATURES:
            raise ValueError(
                f"RBFN requires {self.EXPECTED_OUT_FEATURES} output features, "
                f"but received {out_features}."
            )

        if num_centers <= 0:
            raise ValueError("num_centers must be greater than zero.")

        if rbf_chunk_size <= 0:
            raise ValueError("rbf_chunk_size must be greater than zero.")

        self.in_features = in_features
        self.num_centers = num_centers
        self.out_features = out_features
        self.rbf_chunk_size = rbf_chunk_size

        self.register_buffer(
            "centers",
            torch.empty(
                (0, self.in_features),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "gamma",
            torch.tensor(
                float("nan"),
                dtype=torch.float32,
            ),
        )

        self.readout = nn.Linear(
            self.num_centers,
            self.out_features,
            bias=True,
        )

        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def _validate_input_tensor(
        self,
        X: torch.Tensor,
    ) -> None:
        if not isinstance(X, torch.Tensor):
            raise TypeError(
                f"X must be a torch.Tensor, got {type(X).__name__}."
            )

        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2D tensor of shape (N, {self.in_features}), "
                f"got shape {tuple(X.shape)}."
            )

        if X.shape[1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, got {X.shape[1]}."
            )

        if not torch.is_floating_point(X):
            raise TypeError(
                f"X must use a floating-point dtype, got {X.dtype}."
            )

        if not torch.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinite values. The RBFN requires finite inputs."
            )

        previous_available = X[:, self.PREV_AVAILABLE_INDEX]
        next_available = X[:, self.NEXT_AVAILABLE_INDEX]

        previous_valid = (previous_available == 0.0) | (previous_available == 1.0)
        next_valid = (next_available == 0.0) | (next_available == 1.0)

        if not torch.all(previous_valid):
            raise ValueError(
                "Previous Landsat availability flag must contain only 0 or 1."
            )

        if not torch.all(next_valid):
            raise ValueError(
                "Next Landsat availability flag must contain only 0 or 1."
            )

    def _validate_fitted_state(self) -> None:
        expected_shape = (self.num_centers, self.in_features)

        if self.centers.ndim != 2:
            raise RuntimeError("RBF centres have not been fitted correctly.")

        if tuple(self.centers.shape) != expected_shape:
            raise RuntimeError(
                f"Expected centres with shape {expected_shape}, "
                f"got {tuple(self.centers.shape)}."
            )

        if not torch.isfinite(self.centers).all():
            raise RuntimeError("RBF centres contain NaN or infinite values.")

        if self.gamma.ndim != 0:
            raise RuntimeError("RBF gamma must be a scalar.")

        if not torch.isfinite(self.gamma):
            raise RuntimeError("RBF gamma has not been fitted or is not finite.")

        if self.gamma.item() <= 0.0:
            raise RuntimeError("RBF gamma must be greater than zero.")

    @torch.no_grad()
    def fit_centers_and_gamma(
        self,
        X: np.ndarray | torch.Tensor,
        random_state: int = 42,
        gamma: Optional[float] = None,
        gamma_multiplier: float = 1.0,
    ) -> None:
        """Fit Gaussian RBF centres using K-Means and determine gamma."""
        if gamma is not None and gamma <= 0.0:
            raise ValueError("gamma must be greater than zero.")

        if gamma_multiplier <= 0.0:
            raise ValueError("gamma_multiplier must be greater than zero.")

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        elif isinstance(X, np.ndarray):
            X_np = X
        else:
            raise TypeError("X must be either a numpy.ndarray or torch.Tensor.")

        if X_np.shape[0] < self.num_centers:
            raise ValueError(
                f"Cannot fit {self.num_centers} RBF centres from only "
                f"{X_np.shape[0]} samples."
            )

        if not np.isfinite(X_np).all():
            raise ValueError("Features contain NaN or infinite values.")

        kmeans = KMeans(
            n_clusters=self.num_centers,
            random_state=random_state,
            n_init=10,
        )

        kmeans.fit(X_np)

        model_device = self.readout.weight.device
        model_dtype = self.readout.weight.dtype

        centers = torch.as_tensor(
            kmeans.cluster_centers_,
            dtype=model_dtype,
            device=model_device,
        )

        if gamma is None:
            if self.num_centers < 2:
                raise ValueError(
                    "At least two RBF centres are required to estimate gamma."
                )

            pairwise_distances = torch.cdist(centers, centers, p=2)
            upper_triangle = torch.triu(pairwise_distances, diagonal=1)
            nonzero_distances = upper_triangle[upper_triangle > 0]

            if nonzero_distances.numel() == 0:
                raise ValueError("Unable to estimate gamma because all RBF centres are identical.")

            sigma = torch.median(nonzero_distances).item()

            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("Estimated RBF sigma is invalid.")

            gamma_value = gamma_multiplier / (2.0 * sigma * sigma)
        else:
            gamma_value = float(gamma)

        if not np.isfinite(gamma_value) or gamma_value <= 0.0:
            raise ValueError(f"Invalid RBF gamma: {gamma_value}")

        self.centers = centers
        self.gamma = torch.tensor(
            gamma_value,
            dtype=model_dtype,
            device=model_device,
        )

    def _rbf_chunk(self, X_chunk: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian RBF activations for one input chunk."""
        centers = self.centers.to(device=X_chunk.device, dtype=X_chunk.dtype)
        gamma = self.gamma.to(device=X_chunk.device, dtype=X_chunk.dtype)

        x_squared = torch.sum(X_chunk * X_chunk, dim=1, keepdim=True)
        centers_squared = torch.sum(centers * centers, dim=1, keepdim=True).transpose(0, 1)
        cross_term = torch.matmul(X_chunk, centers.transpose(0, 1))

        distances_squared = x_squared + centers_squared - 2.0 * cross_term
        distances_squared = torch.clamp(distances_squared, min=0.0)

        return torch.exp(-gamma * distances_squared)

    def get_rbf_features(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute RBF activations directly on standardized/prepared input tensors.
        """
        self._validate_fitted_state()
        return self._rbf_chunk(X)

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the RBFN forward pass."""
        phi = self.get_rbf_features(X)
        return self.readout(phi)

    @torch.no_grad()
    def set_readout_parameters(
        self,
        coefficients: np.ndarray | torch.Tensor,
        intercept: np.ndarray | torch.Tensor,
    ) -> None:
        """Load externally fitted Ridge regression parameters."""
        if isinstance(coefficients, np.ndarray):
            coefficients = torch.as_tensor(
                coefficients,
                dtype=self.readout.weight.dtype,
                device=self.readout.weight.device,
            )
        elif isinstance(coefficients, torch.Tensor):
            coefficients = coefficients.to(
                device=self.readout.weight.device,
                dtype=self.readout.weight.dtype,
            )
        else:
            raise TypeError("coefficients must be a numpy.ndarray or torch.Tensor.")

        if isinstance(intercept, np.ndarray):
            intercept = torch.as_tensor(
                intercept,
                dtype=self.readout.bias.dtype,
                device=self.readout.bias.device,
            )
        elif isinstance(intercept, torch.Tensor):
            intercept = intercept.to(
                device=self.readout.bias.device,
                dtype=self.readout.bias.dtype,
            )
        else:
            raise TypeError("intercept must be a numpy.ndarray or torch.Tensor.")

        expected_weight_shape = (self.out_features, self.num_centers)
        expected_bias_shape = (self.out_features,)

        if tuple(coefficients.shape) != expected_weight_shape:
            raise ValueError(
                f"Expected coefficients with shape {expected_weight_shape}, "
                f"got {tuple(coefficients.shape)}."
            )

        if tuple(intercept.shape) != expected_bias_shape:
            raise ValueError(
                f"Expected intercept with shape {expected_bias_shape}, "
                f"got {tuple(intercept.shape)}."
            )

        if not torch.isfinite(coefficients).all():
            raise ValueError("Ridge coefficients contain NaN or infinite values.")

        if not torch.isfinite(intercept).all():
            raise ValueError("Ridge intercept contains NaN or infinite values.")

        self.readout.weight.copy_(coefficients)
        self.readout.bias.copy_(intercept)