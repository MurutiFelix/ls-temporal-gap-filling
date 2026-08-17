# src/predict.py
"""
Out-of-Sample Historical Forward Inference Engine.
Reconstructs Landsat gap months using DEM + time features + ERA5 predictors,
then evaluates derived NDVI against AVHRR NDVI (evaluation-only, not an input).
"""

from pathlib import Path
import numpy as np
import torch
import yaml

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

    checkpoint_path = (
        ROOT_DIR
        / config["paths"]["processed_dir"]
        / "models"
        / "rbfn_landsat_gap_filler.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Execute 'python -m src.train' first."
        )

    processor = RasterProcessor(config)
    n_bands = config["landsat"]["num_bands"]

    gap_start, gap_end = config["landsat"]["gap_years"]

    # Determine target raster shape from DEM (always available)
    dem_probe_path = ROOT_DIR / config["paths"]["static_dir"] / "dem.tif"
    import rasterio
    with rasterio.open(dem_probe_path) as src:
        target_shape = (src.height, src.width)

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
            h, w = target_shape
            time_feat = build_time_features(year, month)
            time_grid = np.tile(time_feat, (h, w, 1))

            n_pixels = h * w
            X_infer_raw = np.concatenate(
                [
                    dem_cube.reshape(n_pixels, -1),
                    time_grid.reshape(n_pixels, -1),
                    era5_cube.reshape(n_pixels, -1),
                ],
                axis=1,
            )
            X_infer_t = torch.tensor(X_infer_raw, dtype=torch.float32)

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

            # Derive indices from the predicted bands
            indices = compute_spectral_indices(predicted_cube)

            # Evaluation-only check against AVHRR NDVI, where available
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

            # TODO: write predicted_cube to a GeoTIFF in data/processed/,
            # flagged as interpolated per the earlier decision on NDVI provenance


if __name__ == "__main__":
    main()