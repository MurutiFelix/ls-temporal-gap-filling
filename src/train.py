# src/train.py
"""
Root Execution Orchestrator.
Loads real Landsat/ERA5/DEM data, trains RBFN on months with Landsat coverage,
holds out a temporal validation split, and saves the model checkpoint.
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


def build_time_features(year: int, month: int) -> np.ndarray:
    """Cyclical month encoding + linear year trend, per features.inputs in config."""
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    year_trend = year  # normalized later alongside other features
    return np.array([month_sin, month_cos, year_trend], dtype=np.float32)


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print(f"[Train Pipeline] Project: {config['project']['name']}")

    processor = RasterProcessor(config)
    training_start, training_end = config["landsat"]["training_years"]

    X_rows = []
    Y_rows = []

    print(f"[Train Pipeline] Scanning {training_start}-{training_end} for Landsat-covered months...")
    for year in range(training_start, training_end + 1):
        for month in range(1, 13):
            landsat_cube = processor.load_landsat_month(year, month)
            if landsat_cube is None:
                continue  # gap month - not usable for training targets

            target_shape = landsat_cube.shape[:2]

            era5_cube = processor.load_era5_month(year, month, target_shape)
            if era5_cube is None:
                continue  # need ERA5 present too, since it's a required input

            dem_cube = processor.load_static(target_shape)

            h, w = target_shape
            time_feat = build_time_features(year, month)
            time_grid = np.tile(time_feat, (h, w, 1))

            # Flatten all pixels for this month into rows
            n_pixels = h * w
            X_month = np.concatenate(
                [
                    dem_cube.reshape(n_pixels, -1),
                    time_grid.reshape(n_pixels, -1),
                    era5_cube.reshape(n_pixels, -1),
                ],
                axis=1,
            )
            Y_month = landsat_cube.reshape(n_pixels, -1)

            X_rows.append(X_month)
            Y_rows.append(Y_month)

    if not X_rows:
        raise RuntimeError(
            "No training months found with both Landsat and ERA5 coverage. "
            "Check data/landsat and data/era5 directories."
        )

    X_raw = np.vstack(X_rows)
    Y_raw = np.vstack(Y_rows)
    print(f"  ✓ Assembled training matrix: X={X_raw.shape}, Y={Y_raw.shape}")

    # Normalize reflectance targets to 0-1
    Y_raw = processor.normalize_reflectance(Y_raw)

    # Temporal holdout split (~22% validation, per config)
    val_ratio = config["model"]["rbfn"]["validation_holdout_ratio"]
    n_samples = X_raw.shape[0]
    split_idx = int(n_samples * (1.0 - val_ratio))

    X_train_t = torch.tensor(X_raw[:split_idx], dtype=torch.float32)
    Y_train_t = torch.tensor(Y_raw[:split_idx], dtype=torch.float32)
    X_val_t = torch.tensor(X_raw[split_idx:], dtype=torch.float32)
    Y_val_t = torch.tensor(Y_raw[split_idx:], dtype=torch.float32)

    model = MultiOutputRBFN(
        in_features=X_train_t.shape[1],
        num_centers=config["model"]["rbfn"]["num_centers"],
        out_bands=config["landsat"]["num_bands"],
    )

    trainer = RBFNTrainer(model, config)
    trainer.fit_ridge(
        X_train_t,
        Y_train_t,
        lambda_reg=config["model"]["rbfn"]["regularization_lambda"],
    )

    val_rmse = trainer.evaluate(X_val_t, Y_val_t)
    print(f"  ✓ Validation Overall RMSE: {val_rmse:.5f}")

    checkpoint_dir = ROOT_DIR / config["paths"]["processed_dir"] / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "rbfn_landsat_gap_filler.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  ✓ Model Checkpoint saved: {model_path}")


if __name__ == "__main__":
    main()