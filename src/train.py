# src/train.py
"""
RBFN Gap-Fill Pipeline — Root Execution Orchestrator.
Loads real Landsat/ERA5/DEM/coordinate data, drops cloud-masked/nodata pixels,
subsamples pixels per month to bound memory usage, standardizes features,
trains RBFN on months with Landsat coverage, holds out a temporal validation
split, and saves the checkpoint + scaler.
"""

from pathlib import Path
import numpy as np
import torch
import yaml
import joblib

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
    year_trend = year
    return np.array([month_sin, month_cos, year_trend], dtype=np.float32)


def drop_invalid_rows(X_month: np.ndarray, Y_month: np.ndarray):
    """Drops any pixel row where X or Y contains NaN or Inf values."""
    valid_mask = np.isfinite(X_month).all(axis=1) & np.isfinite(Y_month).all(axis=1)
    return X_month[valid_mask], Y_month[valid_mask]


def subsample_pixels(X_month: np.ndarray, Y_month: np.ndarray, n_pixels: int, rng: np.random.Generator):
    """Randomly subsamples n_pixels rows from a month's data, without replacement."""
    total = X_month.shape[0]
    if total <= n_pixels:
        return X_month, Y_month  # month already smaller than the cap
    idx = rng.choice(total, size=n_pixels, replace=False)
    return X_month[idx], Y_month[idx]


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  RBFN Gap-Fill Pipeline")
    print("=" * 60)
    print(f"[Train Pipeline] Project: {config['project']['name']}")

    processor = RasterProcessor(config)
    training_start, training_end = config["landsat"]["training_years"]
    pixels_per_month = config["model"]["rbfn"].get("pixels_per_month", 50000)
    rng = np.random.default_rng(config["project"]["seed"])

    X_rows = []
    Y_rows = []

    total_pixels_seen = 0
    total_pixels_after_clean = 0
    total_pixels_kept = 0
    months_used = 0
    months_fully_masked = 0

    print(f"[Train Pipeline] Scanning {training_start}-{training_end} for Landsat-covered months...")
    print(f"[Train Pipeline] Subsampling up to {pixels_per_month} pixels/month to bound memory.")

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
            coord_cube = processor.get_pixel_coords(target_shape)

            h, w = target_shape
            time_feat = build_time_features(year, month)
            time_grid = np.tile(time_feat, (h, w, 1))

            n_pixels = h * w
            X_month = np.concatenate(
                [
                    dem_cube.reshape(n_pixels, -1),
                    coord_cube.reshape(n_pixels, -1),
                    time_grid.reshape(n_pixels, -1),
                    era5_cube.reshape(n_pixels, -1),
                ],
                axis=1,
            )
            Y_month = landsat_cube.reshape(n_pixels, -1)

            total_pixels_seen += n_pixels

            # Drop cloud-masked/nodata/Inf pixels before KMeans or scaling touches data
            X_month, Y_month = drop_invalid_rows(X_month, Y_month)
            total_pixels_after_clean += X_month.shape[0]

            if X_month.shape[0] == 0:
                months_fully_masked += 1
                continue  # entire month was masked out

            X_month, Y_month = subsample_pixels(X_month, Y_month, pixels_per_month, rng)
            total_pixels_kept += X_month.shape[0]
            months_used += 1

            X_rows.append(X_month)
            Y_rows.append(Y_month)

    if not X_rows:
        raise RuntimeError(
            "No usable training months found (Landsat + ERA5 coverage, non-fully-masked). "
            "Check data/landsat and data/era5 directories."
        )

    X_raw = np.vstack(X_rows)
    Y_raw = np.vstack(Y_rows)

    print(f"  ✓ Months used: {months_used} | fully cloud-masked: {months_fully_masked}")
    print(f"  ✓ Pixels seen: {total_pixels_seen:,} | after NaN/Inf drop: {total_pixels_after_clean:,} "
          f"({100 * total_pixels_after_clean / max(total_pixels_seen, 1):.1f}% valid)")
    print(f"  ✓ Pixels kept after subsampling: {total_pixels_kept:,}")
    print(f"  ✓ Assembled training matrix: X={X_raw.shape}, Y={Y_raw.shape}")

    Y_raw = processor.normalize_reflectance(Y_raw)

    processor.fit_scaler(X_raw)
    X_scaled = processor.transform_features(X_raw)

    val_ratio = config["model"]["rbfn"]["validation_holdout_ratio"]
    n_samples = X_scaled.shape[0]
    split_idx = int(n_samples * (1.0 - val_ratio))

    X_train_t = torch.tensor(X_scaled[:split_idx], dtype=torch.float32)
    Y_train_t = torch.tensor(Y_raw[:split_idx], dtype=torch.float32)
    X_val_t = torch.tensor(X_scaled[split_idx:], dtype=torch.float32)
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

    with torch.no_grad():
        val_preds = model(X_val_t).numpy()
    per_band_rmse = compute_rmse(Y_val_t.numpy(), val_preds)
    band_names = config["landsat"]["bands"]
    for name, rmse in zip(band_names, per_band_rmse):
        print(f"    {name}: RMSE={rmse:.5f}")

    checkpoint_dir = ROOT_DIR / config["paths"]["processed_dir"] / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / "rbfn_landsat_gap_filler.pt"
    torch.save(model.state_dict(), model_path)

    scaler_path = checkpoint_dir / "feature_scaler.joblib"
    joblib.dump(processor.scaler, scaler_path)

    print(f"  ✓ Model Checkpoint saved: {model_path}")
    print(f"  ✓ Feature scaler saved: {scaler_path}")


if __name__ == "__main__":
    main()