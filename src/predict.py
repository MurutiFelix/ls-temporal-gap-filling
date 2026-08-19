# src/predict.py
"""
Out-of-Sample Historical Forward Inference Engine.
Reconstructs Landsat gap months using DEM + pixel coordinates + time features +
ERA5 predictors, writes predicted bands to GeoTIFF, then evaluates derived NDVI
against AVHRR NDVI (evaluation-only, not an input).
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
from src.train import build_time_features

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CONFIG_PATH = SRC_DIR / "config.yaml"


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print("[Inference Engine] Loading RBFN checkpoint for gap-month filling...")

    checkpoint_dir = ROOT_DIR / config["paths"]["processed_dir"] / "models"
    checkpoint_path = checkpoint_dir / "rbfn_landsat_gap_filler.pt"
    scaler_path = checkpoint_dir / "feature_scaler.joblib"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Execute 'python -m src.train' first."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Feature scaler not found at {scaler_path}. Execute 'python -m src.train' first."
        )

    processor = RasterProcessor(config)
    processor.scaler = joblib.load(scaler_path)
    processor._scaler_fitted = True

    n_bands = config["landsat"]["num_bands"]
    band_names = config["landsat"]["bands"]
    gap_start, gap_end = config["landsat"]["gap_years"]

    # Determine target raster shape + georeferencing from DEM (always available)
    dem_path = ROOT_DIR / config["paths"]["static_dir"] / "dem.tif"
    with rasterio.open(dem_path) as ref:
        target_shape = (ref.height, ref.width)
        out_meta = ref.meta.copy()

    output_dir = ROOT_DIR / config["paths"]["processed_dir"] / "gapfilled"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = None

    for year in range(gap_start, gap_end + 1):
        for month in range(1, 13):
            landsat_cube = processor.load_landsat_month(year, month)
            if landsat_cube is not None:
                continue  # not a gap month - Landsat already covers this

            era5_cube = processor.load_era5_month(year, month, target_shape)
            if era5_cube is None:
                print(f"  ⚠ Skipping {year}-{month:02d}: no ERA5 coverage either")
                continue

            dem_cube = processor.load_static(target_shape)
            coord_cube = processor.get_pixel_coords(target_shape)
            h, w = target_shape
            time_feat = build_time_features(year, month)
            time_grid = np.tile(time_feat, (h, w, 1))

            n_pixels = h * w
            X_infer_raw = np.concatenate(
                [
                    dem_cube.reshape(n_pixels, -1),
                    coord_cube.reshape(n_pixels, -1),
                    time_grid.reshape(n_pixels, -1),
                    era5_cube.reshape(n_pixels, -1),
                ],
                axis=1,
            )
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
            print(f"  ✓ Reconstructed {year}-{month:02d}: shape {predicted_cube.shape}")

            # ----- Write predicted bands to GeoTIFF, flagged as interpolated -----
            out_meta.update(count=1, dtype="float32")
            for i, band_name in enumerate(band_names):
                out_path = output_dir / f"{band_name.lower()}_{year}_{month:02d}_rbfn_interpolated.tif"
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(predicted_cube[..., i].astype("float32"), 1)

            # ----- Derive indices from the predicted bands -----
            indices = compute_spectral_indices(predicted_cube)

            # ----- Evaluation-only check against AVHRR NDVI, where available -----
            avhrr_ndvi = processor.load_avhrr_month(year, month, target_shape)
            if avhrr_ndvi is not None:
                pred_ndvi_mean = np.nanmean(indices["NDVI"])
                avhrr_ndvi_mean = np.nanmean(avhrr_ndvi)
                print(
                    f"    predicted NDVI mean={pred_ndvi_mean:.4f} vs "
                    f"AVHRR NDVI mean={avhrr_ndvi_mean:.4f} "
                    f"(diff={abs(pred_ndvi_mean - avhrr_ndvi_mean):.4f})"
                )
            else:
                print(f"    (no AVHRR reference available for {year}-{month:02d})")

    print(f"\n  ✓ All gap-filled GeoTIFFs written to: {output_dir}")


if __name__ == "__main__":
    main()