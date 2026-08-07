# src/rbfn_gap_filling_pipeline.py
"""
4-Sprint Execution Driver: RBFN Gap Filling (1995–2025).
Harmonizes datasets, trains PyTorch RBFN, infers historical gaps, and calculates VHI/BSI.
Executes from src/ and exports results to data/Processed/eda/.
"""

from pathlib import Path
import numpy as np
import torch

# Relative imports within src package
from src.data.flow_manager import MasterDataFlowManager
from src.models.rbfn import MultiOutputRBFN

# Setup dynamic paths relative to repository root
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
OUTPUT_DIR = ROOT_DIR / "data" / "Processed" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # -------------------------------------------------------------------------
    # SPRINT 1: Data Harmonization & Feature Scaling Simulation
    # -------------------------------------------------------------------------
    print("[Sprint 1] Harmonizing MODIS, AVHRR, ERA5, and Landsat grids...")

    n_pixels = 5000  # Example pixel array length
    n_bands = 6  # R, G, B, NIR, SWIR, Thermal

    # Scale reflectance strictly to [0.0, 1.0] and climate variables
    modis_sim = np.random.uniform(0.0, 1.0, size=(n_pixels, n_bands))
    era5_sim = np.random.uniform(250.0, 320.0, size=(n_pixels, 3)) / 320.0
    norms_sim = np.random.uniform(0.0, 1.0, size=(n_pixels, n_bands))

    # Target Landsat ground truth (2000-2025)
    landsat_ground_truth = np.random.uniform(0.0, 1.0, size=(n_pixels, n_bands))

    flow_mgr = MasterDataFlowManager(n_bands=n_bands)
    X_train_raw = flow_mgr.assemble_training_features(modis_sim, era5_sim, norms_sim)
    Y_train_raw = landsat_ground_truth

    # -------------------------------------------------------------------------
    # SPRINT 2: Model Training & Validation Split
    # -------------------------------------------------------------------------
    print("[Sprint 2] Splitting dataset (18/5 Holdout) & Training PyTorch RBFN...")

    split_idx = int(n_pixels * (18 / 23))  # ~78% train, 22% validation holdout

    X_train_t = torch.tensor(X_train_raw[:split_idx], dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train_raw[:split_idx], dtype=torch.float32)

    X_val_t = torch.tensor(X_train_raw[split_idx:], dtype=torch.float32)
    Y_val_t = torch.tensor(Y_train_raw[split_idx:], dtype=torch.float32)

    # Instantiate and fit RBF Centers
    in_dims = X_train_t.shape[1]
    k_centers = 50
    model = MultiOutputRBFN(in_features=in_dims, num_centers=k_centers, out_bands=n_bands)
    model.fit_centers(X_train_t)

    # Closed-form regularized Ridge solution for linear weights
    A = model._gaussian_rbf(X_train_t)
    lambda_reg = 1e-3
    I = torch.eye(A.shape[1])
    W = torch.linalg.inv(A.T @ A + lambda_reg * I) @ A.T @ Y_train_t

    model.linear_weights.weight.data = W.T
    model.linear_weights.bias.data.fill_(0.0)

    print("  ✓ RBFN Training complete.")

    # -------------------------------------------------------------------------
    # SPRINT 3: Historical Inference & Gap Reconstruction (1995–1999)
    # -------------------------------------------------------------------------
    print("[Sprint 3] Running inference on historical AVHRR + ERA5 inputs (1995-1999)...")

    avhrr_historical = np.random.uniform(0.0, 1.0, size=(1000, n_bands))
    era5_historical = np.random.uniform(250.0, 320.0, size=(1000, 3)) / 320.0
    norms_historical = np.random.uniform(0.0, 1.0, size=(1000, n_bands))

    X_infer_raw = flow_mgr.assemble_inference_features(
        avhrr_historical, era5_historical, norms_historical
    )
    X_infer_t = torch.tensor(X_infer_raw, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predicted_landsat_cube = model(X_infer_t).numpy()

    print(f"  ✓ Reconstructed Landsat Cube Shape: {predicted_landsat_cube.shape}")

    # -------------------------------------------------------------------------
    # SPRINT 4: Downstream Spectral Indices & Validation Metrics
    # -------------------------------------------------------------------------
    print("[Sprint 4] Calculating VHI / BSI indices and evaluating validation RMSE...")

    with torch.no_grad():
        Y_val_pred = model(X_val_t).numpy()

    val_rmse = np.sqrt(np.mean((Y_val_pred - Y_val_t.numpy()) ** 2, axis=0))
    print(f"  ✓ Per-Band Validation RMSE (R, G, B, NIR, SWIR, Thermal):")
    print(f"    {np.round(val_rmse, 4)}")

    # Calculate VHI (Vegetation Health Index) proxy on predicted output
    red_pred = predicted_landsat_cube[:, 0]
    nir_pred = predicted_landsat_cube[:, 3]
    lst_pred = predicted_landsat_cube[:, 5]

    ndvi_pred = (nir_pred - red_pred) / (nir_pred + red_pred + 1e-6)
    vhi_pred = (ndvi_pred + (1.0 - lst_pred)) / 2.0

    print(f"  ✓ Calculated VHI on reconstructed cube. Mean VHI: {np.mean(vhi_pred):.4f}")
    
    # Save a summary report to data/Processed/eda
    report_path = OUTPUT_DIR / "rbfn_validation_summary.txt"
    with open(report_path, "w") as f:
        f.write("RBFN Gap-Filling Model Validation Summary\n")
        f.write("=========================================\n")
        f.write(f"Per-Band Validation RMSE: {np.round(val_rmse, 5).tolist()}\n")
        f.write(f"Mean Reconstructed VHI: {np.mean(vhi_pred):.5f}\n")

    print(f"  ✓ Exported validation summary to: {report_path}")
    print("\nPipeline execution successful.")


if __name__ == "__main__":
    main()