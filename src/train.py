# src/train.py
"""
Root Execution Orchestrator.
Imports from updated preprocessing package: src.preprocessing.raster_processor
"""

from pathlib import Path
import numpy as np
import torch
import yaml

from src.models.rbfn import MultiOutputRBFN
from src.models.train_rbfn import RBFNTrainer
from src.preprocessing.raster_processor import RasterProcessor
from src.utils.metrics import compute_rmse

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CONFIG_PATH = SRC_DIR / "config.yaml"


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print(f"[Train Pipeline] Project: {config['project']['name']}")

    processor = RasterProcessor(config)
    n_pixels = 5000
    n_bands = config["landsat"]["num_bands"]

    # Simulating data ingestion from data/modis, data/avhrr, data/static
    modis_sim = processor.normalize_reflectance(
        np.random.uniform(0.0, 1.0, size=(n_pixels, n_bands))
    )
    avhrr_sim = processor.normalize_reflectance(
        np.random.uniform(0.0, 1.0, size=(n_pixels, 1))
    )  # NDVI
    static_sim = np.random.uniform(0.0, 1.0, size=(n_pixels, 2))  # DEM, Soil
    landsat_target = np.random.uniform(0.0, 1.0, size=(n_pixels, n_bands))

    X_raw = processor.assemble_feature_matrix(modis_sim, avhrr_sim, static_sim)

    # 18/5 temporal holdout split (~22% validation)
    val_ratio = config["model"]["rbfn"]["validation_holdout_ratio"]
    split_idx = int(n_pixels * (1.0 - val_ratio))

    X_train_t = torch.tensor(X_raw[:split_idx], dtype=torch.float32)
    Y_train_t = torch.tensor(landsat_target[:split_idx], dtype=torch.float32)

    X_val_t = torch.tensor(X_raw[split_idx:], dtype=torch.float32)
    Y_val_t = landsat_target[split_idx:]

    # Initialize and train model via src/models/train_rbfn.py
    model = MultiOutputRBFN(
        in_features=X_train_t.shape[1],
        num_centers=config["model"]["rbfn"]["num_centers"],
        out_bands=n_bands,
    )

    trainer = RBFNTrainer(model, config)
    trainer.fit_ridge(
        X_train_t,
        Y_train_t,
        lambda_reg=config["model"]["rbfn"]["regularization_lambda"],
    )

    # Evaluate validation accuracy
    val_rmse = trainer.evaluate(X_val_t, torch.tensor(Y_val_t, dtype=torch.float32))
    print(f"  ✓ Validation Overall RMSE: {val_rmse:.5f}")

    # Checkpoint saving
    checkpoint_dir = ROOT_DIR / config["paths"]["processed_dir"] / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "rbfn_landsat_gap_filler.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  ✓ Model Checkpoint saved: {model_path}")


if __name__ == "__main__":
    main()