# src/train.py
"""
Root Execution Orchestrator for Landsat RBFN Model Training.
Reads src/config.yaml, initializes PyTorch RBFN, fits centers, and saves checkpoint.
"""

from pathlib import Path
import numpy as np
import torch
import yaml

from src.data.raster_processor import RasterProcessor
from src.models.rbfn import MultiOutputRBFN
from src.utils.metrics import compute_reconstruction_rmse

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CONFIG_PATH = SRC_DIR / "config.yaml"


def main():
    # 1. Load Centralized Configuration
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print(f"[Train Orchestrator] Project: {config['project']['name']}")

    # 2. Prepare Data Arrays via RasterProcessor
    processor = RasterProcessor(config)
    n_pixels = 5000
    n_bands = config["landsat"]["num_bands"]

    modis_sim = processor.normalize_reflectance(
        np.random.uniform(0.0, 1.0, size=(n_pixels, n_bands))
    )
    era5_sim = (
        np.random.uniform(250.0, 320.0, size=(n_pixels, 3)) / 320.0
    )  # Kelvin scaling
    norms_sim = np.random.uniform(0.0, 1.0, size=(n_pixels, n_bands))
    landsat_target = np.random.uniform(0.0, 1.0, size=(n_pixels, n_bands))

    X_raw = processor.assemble_feature_matrix(modis_sim, era5_sim, norms_sim)

    # Train / Validation Holdout Split (18/5 ratio ~ 78% train, 22% val)
    val_ratio = config["model"]["rbfn"]["validation_holdout_ratio"]
    split_idx = int(n_pixels * (1.0 - val_ratio))

    X_train_t = torch.tensor(X_raw[:split_idx], dtype=torch.float32)
    Y_train_t = torch.tensor(landsat_target[:split_idx], dtype=torch.float32)

    X_val_t = torch.tensor(X_raw[split_idx:], dtype=torch.float32)
    Y_val_t = landsat_target[split_idx:]

    # 3. Instantiate and Fit PyTorch RBFN
    model = MultiOutputRBFN(
        in_features=X_train_t.shape[1],
        num_centers=config["model"]["rbfn"]["num_centers"],
        out_bands=n_bands,
    )
    model.fit_centers(X_train_t)

    # Closed-form Ridge Regression for linear weights
    A = model._gaussian_rbf(X_train_t)
    lambda_reg = config["model"]["rbfn"]["regularization_lambda"]
    I = torch.eye(A.shape[1])
    W = torch.linalg.inv(A.T @ A + lambda_reg * I) @ A.T @ Y_train_t

    model.linear_weights.weight.data = W.T
    model.linear_weights.bias.data.fill_(0.0)

    # 4. Evaluate Validation Accuracy
    model.eval()
    with torch.no_grad():
        Y_val_pred = model(X_val_t).numpy()

    val_rmse = compute_reconstruction_rmse(Y_val_t, Y_val_pred)
    print(f"  ✓ Per-Band Validation RMSE (R, G, B, NIR, SWIR, Thermal):")
    print(f"    {np.round(val_rmse, 5)}")

    # 5. Save Model Checkpoint
    checkpoint_dir = ROOT_DIR / config["paths"]["processed_data"] / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "rbfn_landsat_gap_filler.pt"
    torch.save(model.state_dict(), model_path)

    print(f"  ✓ Model training complete. Saved checkpoint to: {model_path}")


if __name__ == "__main__":
    main()