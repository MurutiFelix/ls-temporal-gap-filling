# src/predict.py
"""
Out-of-Sample Historical Forward Inference Engine.
Reconstructs deep historical gaps (1995-1999) using AVHRR and ERA5 predictors.
"""

from pathlib import Path
import numpy as np
import torch
import yaml

from src.preprocessing.raster_processor import RasterProcessor
from src.preprocessing.indices import compute_spectral_indices
from src.models.rbfn import MultiOutputRBFN

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CONFIG_PATH = SRC_DIR / "config.yaml"


def main():
    # 1. Load Configuration & Model Weights
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    print("[Inference Engine] Loading RBFN checkpoint for 1995–1999 gap filling...")

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
    n_pixels = 1000
    n_bands = config["landsat"]["num_bands"]

    # 2. Simulate Historical AVHRR + ERA5 Predictors
    avhrr_hist = np.random.uniform(0.0, 1.0, size=(n_pixels, n_bands))
    era5_hist = np.random.uniform(250.0, 320.0, size=(n_pixels, 3)) / 320.0
    norms_hist = np.random.uniform(0.0, 1.0, size=(n_pixels, n_bands))

    X_infer_raw = processor.assemble_feature_matrix(
        avhrr_hist, era5_hist, norms_hist
    )
    X_infer_t = torch.tensor(X_infer_raw, dtype=torch.float32)

    # 3. Load Trained RBFN
    model = MultiOutputRBFN(
        in_features=X_infer_t.shape[1],
        num_centers=config["model"]["rbfn"]["num_centers"],
        out_bands=n_bands,
    )
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # 4. Predict Missing Landsat Data Cube
    with torch.no_grad():
        predicted_cube = model(X_infer_t).numpy()

    print(f"  ✓ Reconstructed Landsat Cube Shape: {predicted_cube.shape}")

    # 5. Calculate Downstream Indices (VHI / BSI / NDVI)
    indices = compute_spectral_indices(predicted_cube)
    print(f"  ✓ Mean Reconstructed VHI: {np.mean(indices['VHI']):.4f}")
    print(f"  ✓ Mean Reconstructed BSI: {np.mean(indices['BSI']):.4f}")


if __name__ == "__main__":
    main()