# src/evaluate_ndvi_correlation.py

"""
Comprehensive NDVI Cross-Sensor Validation Pipeline.
Evaluates Landsat (Gap-Filled and Compiled Raw) against MODIS and AVHRR benchmarks (1995-2025).
"""

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

ROOT_DIR = Path(__file__).resolve().parents[1]
GAPFILLED_DIR = ROOT_DIR / "outputs" / "gapfilled"
COMPILED_DIR = ROOT_DIR / "outputs" / "compiled"

# Direct paths to reference datasets
MODIS_DIR = ROOT_DIR / "data" / "modis"
AVHRR_DIR = ROOT_DIR / "data" / "avhrr"

RESULTS_CSV = ROOT_DIR / "outputs" / "ndvi_correlation_results.csv"


def load_or_compute_ndvi(tif_path: Path) -> Tuple[np.ndarray, dict]:
    """Reads single-band NDVI or computes NDVI from 7-band Landsat raster."""
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        if src.count == 1:
            ndvi = src.read(1).astype(np.float32)
        else:
            red = src.read(1).astype(np.float32)
            nir = src.read(4).astype(np.float32)
            denom = nir + red
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi = np.where(denom > 1e-4, (nir - red) / denom, np.nan)

        return np.clip(ndvi, -1.0, 1.0), profile


def evaluate_ndvi_pair(
    landsat_path: Path, ref_path: Path, sensor_name: str, scene_type: str
) -> dict:
    """Reprojects reference NDVI to Landsat grid and computes r, RMSE, and SSIM."""
    ls_ndvi, ls_profile = load_or_compute_ndvi(landsat_path)

    with rasterio.open(ref_path) as src_ref:
        ref_raw = src_ref.read(1).astype(np.float32)

        # Scale check for reference integer rasters (e.g., MODIS scaled by 10,000)
        if np.nanmedian(ref_raw[ref_raw > 0]) > 1.0:
            ref_raw = ref_raw / 10000.0

        ref_resampled = np.zeros_like(ls_ndvi, dtype=np.float32)
        reproject(
            source=ref_raw,
            destination=ref_resampled,
            src_transform=src_ref.transform,
            src_crs=src_ref.crs,
            dst_transform=ls_profile["transform"],
            dst_crs=ls_profile["crs"],
            resampling=Resampling.bilinear,
        )

    # Valid mask inside catchment
    valid = (
        np.isfinite(ls_ndvi)
        & np.isfinite(ref_resampled)
        & (ls_ndvi >= -1.0) & (ls_ndvi <= 1.0)
        & (ref_resampled >= -1.0) & (ref_resampled <= 1.0)
    )

    if np.sum(valid) < 100:
        return None

    ls_v = ls_ndvi[valid]
    ref_v = ref_resampled[valid]

    r_val, _ = pearsonr(ls_v, ref_v)
    rmse_val = np.sqrt(np.mean((ls_v - ref_v) ** 2))

    # Fill neutral median for 2D SSIM
    ls_2d = np.where(valid, ls_ndvi, np.nanmedian(ls_v))
    ref_2d = np.where(valid, ref_resampled, np.nanmedian(ref_v))
    ssim_val = ssim(ls_2d, ref_2d, data_range=2.0)

    # Extract Year and Month from filename
    match = re.search(r"_(\d{4})_(\d{1,2})", landsat_path.stem)
    year = int(match.group(1)) if match else None
    month = int(match.group(2)) if match else None

    return {
        "year": year,
        "month": month,
        "landsat_file": landsat_path.name,
        "reference_sensor": sensor_name,
        "scene_type": scene_type,  # 'GapFilled' or 'Compiled_Raw'
        "pearson_r": float(r_val),
        "rmse": float(rmse_val),
        "ssim": float(ssim_val),
        "valid_pixel_count": int(np.sum(valid)),
    }


def build_reference_map(directory: Path, prefix: str) -> dict:
    """
    Indexes reference GeoTIFF files by (year, month) tuple.
    Expects pattern: {prefix}_ndvi_{YYYY}_{MM}.tif (e.g., modis_ndvi_2006_01.tif)
    """
    mapping = {}
    if not directory.exists():
        print(f"Warning: Directory does not exist -> {directory}")
        return mapping

    # Match modis_ndvi_YYYY_MM or avhrr_ndvi_YYYY_MM
    pattern = re.compile(rf"{prefix}_ndvi_(\d{{4}})_(\d{{1,2}})", re.IGNORECASE)

    tif_files = list(directory.glob("*.tif")) + list(directory.glob("*.tiff"))
    for p in tif_files:
        match = pattern.search(p.stem)
        if match:
            y, m = int(match.group(1)), int(match.group(2))
            mapping[(y, m)] = p

    return mapping


def main():
    print("=========================================================")
    print(" Starting Multi-Sensor NDVI Correlation & SSIM Pipeline")
    print(" MODIS Directory: ", MODIS_DIR)
    print(" AVHRR Directory: ", AVHRR_DIR)
    print("=========================================================\n")

    modis_map = build_reference_map(MODIS_DIR, "modis")
    avhrr_map = build_reference_map(AVHRR_DIR, "avhrr")

    print(f"Indexed {len(avhrr_map)} AVHRR reference files (avhrr_ndvi_YYYY_MM.tif).")
    print(f"Indexed {len(modis_map)} MODIS reference files (modis_ndvi_YYYY_MM.tif).\n")

    results = []

    # 1. EVALUATE GAP-FILLED OUTPUTS (Model Reconstructions)
    gapfilled_files = list(GAPFILLED_DIR.glob("*.tif"))
    print(f"Evaluating {len(gapfilled_files)} files in outputs/gapfilled/...")

    for gf_path in gapfilled_files:
        match = re.search(r"_(\d{4})_(\d{1,2})", gf_path.stem)
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))

        # Compare vs AVHRR (1995-2024)
        if key in avhrr_map:
            res = evaluate_ndvi_pair(gf_path, avhrr_map[key], "AVHRR", "GapFilled")
            if res:
                results.append(res)

        # Compare vs MODIS (2003-2025)
        if key in modis_map:
            res = evaluate_ndvi_pair(gf_path, modis_map[key], "MODIS", "GapFilled")
            if res:
                results.append(res)

    # 2. EVALUATE COMPILED RAW SCENES (Clean Baselines >= 80% Coverage)
    compiled_raw_files = list(COMPILED_DIR.glob("*_R.tif"))
    print(f"Evaluating {len(compiled_raw_files)} raw baseline scenes in outputs/compiled/...")

    for raw_path in compiled_raw_files:
        match = re.search(r"_(\d{4})_(\d{1,2})", raw_path.stem)
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))

        if key in avhrr_map:
            res = evaluate_ndvi_pair(raw_path, avhrr_map[key], "AVHRR", "Compiled_Raw")
            if res:
                results.append(res)

        if key in modis_map:
            res = evaluate_ndvi_pair(raw_path, modis_map[key], "MODIS", "Compiled_Raw")
            if res:
                results.append(res)

    # SAVE RESULTS
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_CSV, index=False)
        print(f"\n=========================================================")
        print(f" Successfully saved metric results to: {RESULTS_CSV}")
        print("=========================================================\n")

        print("=== Summary Metrics by Category ===")
        summary = df.groupby(["scene_type", "reference_sensor"])[
            ["pearson_r", "rmse", "ssim"]
        ].mean()
        print(summary.to_string())
    else:
        print("\nNo matching date pairs evaluated. Please verify filenames in data/modis and data/avhrr.")


if __name__ == "__main__":
    main()