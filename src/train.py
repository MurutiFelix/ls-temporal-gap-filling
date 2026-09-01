# src/train.py
"""
Context-Aware RBFN Training Engine.
Uses temporal neighbors, adaptive window deltas, static features, and ERA5 predictors.
"""

from pathlib import Path
import numpy as np
import torch
import yaml
import joblib
import rasterio

from src.models.rbfn import MultiOutputRBFN
from src.models.train_rbfn import RBFNTrainer
from src.preprocessing.raster_processor import RasterProcessor
from src.utils.metrics import compute_rmse
from src.utils.spatial import generate_spatial_coordinates

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CONFIG_PATH = SRC_DIR / "config.yaml"


def build_time_features(year: int, month: int) -> np.ndarray:
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    year_trend = float(year)
    return np.array([month_sin, month_cos, year_trend], dtype=np.float32)


def drop_invalid_rows(X_month: np.ndarray, Y_month: np.ndarray):
    valid_mask = np.isfinite(X_month).all(axis=1) & np.isfinite(Y_month).all(axis=1)
    return X_month[valid_mask], Y_month[valid_mask]


def subsample_pixels(
    X_month: np.ndarray, Y_month: np.ndarray, n_pixels: int, rng: np.random.Generator
):
    total = X_month.shape[0]
    if total <= n_pixels:
        return X_month, Y_month
    idx = rng.choice(total, size=n_pixels, replace=False)
    return X_month[idx], Y_month[idx]


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  Context-Aware RBFN Gap-Fill Training")
    print("=" * 60)

    processor = RasterProcessor(config)
    training_start, training_end = config["landsat"]["training_years"]
    pixels_per_month = config["model"]["rbfn"].get("pixels_per_month", 50000)
    rng = np.random.default_rng(config["project"]["seed"])

    dem_path = ROOT_DIR / config["paths"]["static_dir"] / "dem.tif"
    with rasterio.open(dem_path) as ref:
        bounds = (ref.bounds.left, ref.bounds.bottom, ref.bounds.right, ref.bounds.top)

    X_rows, Y_rows = [], []

    for year in range(training_start, training_end + 1):
        for month in range(1, 13):
            target_cube = processor.load_landsat_month(year, month)
            if target_cube is None:
                continue

            target_shape = target_cube.shape[:2]
            h, w = target_shape

            # Retrieve closest temporal neighbors
            neighbors = processor.load_temporal_neighbors(year, month)
            if neighbors["prev_cube"] is None or neighbors["next_cube"] is None:
                continue  # Skip if isolated without temporal context

            era5_cube = processor.load_era5_month(year, month, target_shape)
            if era5_cube is None:
                continue

            dem_cube = processor.load_static(target_shape)
            coords_2d = generate_spatial_coordinates(target_shape, bounds)
            time_feat = build_time_features(year, month)
            time_grid = np.tile(time_feat, (h * w, 1))

            # Assemble delta time grids
            dt_grid = np.tile(
                np.array([neighbors["dt_prev"], neighbors["dt_next"]], dtype=np.float32),
                (h * w, 1),
            )

            # Flatten feature matrices
            prev_flat = neighbors["prev_cube"].reshape(h * w, -1)
            next_flat = neighbors["next_cube"].reshape(h * w, -1)
            dem_flat = dem_cube.reshape(h * w, -1)
            era5_flat = era5_cube.reshape(h * w, -1)

            # Concatenate into unified context vector
            X_month = np.hstack([
                prev_flat,
                next_flat,
                dt_grid,
                dem_flat,
                coords_2d,
                time_grid,
                era5_flat,
            ])
            Y_month = target_cube.reshape(h * w, -1)

            X_month, Y_month = drop_invalid_rows(X_month, Y_month)
            if X_month.shape[0] == 0:
                continue

            X_month, Y_month = subsample_pixels(X_month, Y_month, pixels_per_month, rng)
            X_rows.append(X_month)
            Y_rows.append(Y_month)

    X_raw = np.vstack(X_rows)
    Y_raw = np.vstack(Y_rows)

    Y_raw = processor.normalize_reflectance(Y_raw)

    val_ratio = config["model"]["rbfn"]["validation_holdout_ratio"]
    split_idx = int(X_raw.shape[0] * (1.0 - val_ratio))

    X_train_raw, X_val_raw = X_raw[:split_idx], X_raw[split_idx:]
    Y_train, Y_val = Y_raw[:split_idx], Y_raw[split_idx:]

    processor.fit_scaler(X_train_raw)
    X_train_scaled = processor.transform_features(X_train_raw)
    X_val_scaled = processor.transform_features(X_val_raw)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)

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
    print(f"  ✓ Context-Aware Validation RMSE: {val_rmse:.5f}")

    checkpoint_dir = ROOT_DIR / config["paths"]["processed_dir"] / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_dir / "rbfn_landsat_gap_filler.pt")
    joblib.dump(processor.scaler, checkpoint_dir / "feature_scaler.joblib")
    print(f"  ✓ Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()