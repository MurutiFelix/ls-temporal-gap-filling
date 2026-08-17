# src/models/rbfn.py
"""
Multi-Output Radial Basis Function Network (RBFN) in PyTorch.
Uses Gaussian kernels for non-linear feature mapping and Ridge regression outputs.
"""

from sklearn.cluster import KMeans
import torch
import torch.nn as nn


class MultiOutputRBFN(nn.Module):
    """
    Multi-Output Radial Basis Function Network for multi-band Landsat gap filling.
    Predicts 7 spectral bands simultaneously (Red, Green, Blue, NIR, SWIR1, SWIR2, Thermal).
    """

    def __init__(
        self,
        in_features: int,
        num_centers: int,
        out_bands: int = 7,
        gamma: float = None,
    ):
        super(MultiOutputRBFN, self).__init__()
        self.in_features = in_features
        self.num_centers = num_centers
        self.out_bands = out_bands

        # Hidden layer centers (c_k)
        self.centers = nn.Parameter(
            torch.Tensor(num_centers, in_features), requires_grad=False
        )
        self.gamma = gamma

        # Output linear weights mapping K centers -> 7 Bands
        self.linear_weights = nn.Linear(num_centers, out_bands, bias=True)

    def fit_centers(self, X: torch.Tensor):
        """Initializes RBF centers using K-Means clustering on feature space."""
        X_np = X.detach().cpu().numpy()
        kmeans = KMeans(
            n_clusters=self.num_centers, random_state=42, n_init=10
        )
        kmeans.fit(X_np)

        self.centers.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        )

        if self.gamma is None:
            dists = torch.cdist(self.centers, self.centers)
            mean_dist = torch.mean(dists)
            self.gamma = 1.0 / (2.0 * (mean_dist**2) + 1e-8)

    def _gaussian_rbf(self, X: torch.Tensor) -> torch.Tensor:
        """Computes Gaussian activation: exp(-gamma * ||x - c_k||^2)"""
        distances = torch.cdist(X, self.centers, p=2)
        return torch.exp(-self.gamma * (distances**2))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass: X -> Gaussian RBF activations -> Linear multi-band outputs."""
        rbf_activations = self._gaussian_rbf(X)
        return self.linear_weights(rbf_activations)