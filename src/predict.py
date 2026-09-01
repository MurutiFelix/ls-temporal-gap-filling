# src/predict.py
"""
Context-Aware Historical Gap-Filling Inference Engine.
Reconstructs missing Landsat months using temporal neighbor windows, adaptive deltas,
DEM, coordinates, time features, and ERA5 predictors. Writes output bands to GeoTIFFs.
"""

from pathlib import Path
import numpy as np
import torch
import yaml
import joblib
import rasterio

from src.preprocessing.raster_processor import RasterProcessor
from src.preprocessing.indices import compute_spectral_indices
from src.models.rbfn import MultiOutputRBFN
from src.utils.spatial import generate_spatial_coordinates
from src.train import build_time_features

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CONFIG_PATH = SRC_DIR / "config.yaml"


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print("[Inference Engine] Loading Context-Aware RBFN checkpoint...")

    checkpoint_dir = ROOT_DIR / config["paths"]["processed_dir"] / "models"
    checkpoint_path = checkpoint_dir / "rbfn_landsat_gap_filler.pt"
    scaler_path = checkpoint_dir / "feature_scaler.joblib"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint missing at {checkpoint_path}. Run 'python -m src.train' first."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler missing at {scaler_path}. Run 'python -m src.train' first."
        )

    processor = RasterProcessor(config)
    processor.scaler = joblib.load(scaler_path)
    processor._scaler_fitted = True

    n_bands = config["landsat"]["num_bands"]
    band_names = config["landsat"]["bands"]
    gap_start, gap_end = config["landsat"]["gap_years"]

    dem_path = ROOT_DIR / config["paths"]["static_dir"] / "dem.tif"
    with rasterio.open(dem_path) as ref:
        target_shape = (ref.height, ref.width)
        out_meta = ref.meta.copy()
        bounds = (ref.bounds.left, ref.bounds.bottom, ref.bounds.right, ref.bounds.top)

    output_dir = ROOT_DIR / config["paths"]["processed_dir"] / "gapfilled"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = None

    for year in range(gap_start, gap_end + 1):
        for month in range(1, 13):
            landsat_cube = processor.load_landsat_month(year, month)
            if landsat_cube is not None:
                continue  # Skip months with existing Landsat coverage

            era5_cube = processor.load_era5_month(year, month, target_shape)
            if era5_cube is None:
                print(f"  ⚠ Skipping {year}-{month:02d}: ERA5 predictors missing")
                continue

            neighbors = processor.load_temporal_neighbors(year, month)
            if neighbors["prev_cube"] is None or neighbors["next_cube"] is None:
                print(f"  ⚠ Skipping {year}-{month:02d}: temporal context unavailable")
                continue

            dem_cube = processor.load_static(target_shape)
            coords_2d = generate_spatial_coordinates(target_shape, bounds)
            h, w = target_shape

            time_feat = build_time_features(year, month)
            time_grid = np.tile(time_feat, (h * w, 1))

            dt_grid = np.tile(
                np.array([neighbors["dt_prev"], neighbors["dt_next"]], dtype=np.float32),
                (h * w, 1),
            )

            prev_flat = neighbors["prev_cube"].reshape(h * w, -1)
            next_flat = neighbors["next_cube"].reshape(h * w, -1)
            dem_flat = dem_cube.reshape(h * w, -1)
            era5_flat = era5_cube.reshape(h * w, -1)

            X_infer_raw = np.hstack([
                prev_flat,
                next_flat,
                dt_grid,
                dem_flat,
                coords_2d,
                time_grid,
                era5_flat,
            ])

            X_infer_scaled = processor.transform_features(X_infer_raw)
            X_infer_t = torch.tensor(X_infer_scaled, dtype=torch.float32)

            if model is None:
                model = MultiOutputRBFN(
                    in_features=X_infer_t.shape[1],
                    num_centers=config["model"]["rbfn"]["num_centers"],
                    out_bands=n_bands,
                )
                model.load_state_dict(torch.load(checkpoint_path))
                model.eval()

            with torch.no_grad():
                predicted_flat = model(X_infer_t).numpy()

            predicted_cube = predicted_flat.reshape(h, w, n_bands)
            print(f"  ✓ Reconstructed Landsat Gap Scene: {year}-{month:02d}")

            # Export predicted bands to GeoTIFFs
            out_meta.update(count=1, dtype="float32")
            for i, band_name in enumerate(band_names):
                out_path = output_dir / f"{band_name.lower()}_{year}_{month:02d}_rbfn_interpolated.tif"
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(predicted_cube[..., i].astype("float32"), 1)

            # Evaluate derived NDVI against AVHRR reference (if present)
            indices = compute_spectral_indices(predicted_cube)
            avhrr_ndvi = processor.load_avhrr_month(year, month, target_shape)
            if avhrr_ndvi is not None:
                pred_ndvi_mean = np.nanmean(indices["NDVI"])
                avhrr_ndvi_mean = np.nanmean(avhrr_ndvi)
                print(
                    f"    NDVI Check -> Pred Mean: {pred_ndvi_mean:.4f} | "
                    f"AVHRR Mean: {avhrr_ndvi_mean:.4f} "
                    f"(Diff: {abs(pred_ndvi_mean - avhrr_ndvi_mean):.4f})"
                )

    print(f"\n  ✓ Gap-filling complete! Exports saved to: {output_dir}")


if __name__ == "__main__":
    main()