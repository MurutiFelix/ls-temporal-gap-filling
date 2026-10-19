# src/evaluate_ndvi_correlation.py

"""
Evaluates compiled Landsat NDVI outputs against MODIS/AVHRR reference imagery.
Computes Pearson Correlation (r), RMSE, and Structural Similarity (SSIM).
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

ROOT_DIR = Path(__file__).resolve().parents[1]
LANDSAT_NDVI_DIR = ROOT_DIR / "outputs" / "compiled"
REFERENCE_NDVI_DIR = ROOT_DIR / "data" / "reference_ndvi"  # MODIS or AVHRR directory


def evaluate_ndvi_pair(landsat_path: Path, reference_path: Path) -> dict:
    with rasterio.open(landsat_path) as src_ls, rasterio.open(reference_path) as src_ref:
        ls_ndvi = src_ls.read(1)
        ref_raw = src_ref.read(1)

        # Reproject reference raster to match exact Landsat grid
        ref_resampled = np.zeros_like(ls_ndvi, dtype=np.float32)
        reproject(
            source=ref_raw,
            destination=ref_resampled,
            src_transform=src_ref.transform,
            src_crs=src_ref.crs,
            dst_transform=src_ls.transform,
            dst_crs=src_ls.crs,
            resampling=Resampling.bilinear,
        )

        # Create valid mask inside AOI (NDVI strictly between -1.0 and 1.0)
        valid = (
            np.isfinite(ls_ndvi)
            & np.isfinite(ref_resampled)
            & (ls_ndvi >= -1.0) & (ls_ndvi <= 1.0)
            & (ref_resampled >= -1.0) & (ref_resampled <= 1.0)
        )

        if np.sum(valid) < 100:
            return None

        ls_valid = ls_ndvi[valid]
        ref_valid = ref_resampled[valid]

        # 1. Pearson Correlation Coefficient (r)
        r_val, _ = pearsonr(ls_valid, ref_valid)

        # 2. Root Mean Squared Error (RMSE)
        rmse_val = np.sqrt(np.mean((ls_valid - ref_valid) ** 2))

        # 3. 2D Structural Similarity Index (SSIM)
        # Fill non-valid pixels with neutral median for 2D SSIM calculation
        ls_2d = np.where(valid, ls_ndvi, np.nanmedian(ls_valid))
        ref_2d = np.where(valid, ref_resampled, np.nanmedian(ref_valid))
        ssim_val = ssim(ls_2d, ref_2d, data_range=2.0)

        return {
            "pearson_r": float(r_val),
            "rmse": float(rmse_val),
            "ssim": float(ssim_val),
            "pixel_count": int(np.sum(valid)),
        }


if __name__ == "__main__":
    print("Running Landsat vs MODIS/AVHRR NDVI correlation...")
    # Add loop over matching dates here once reference files are placed in data/reference_ndvi/