# src/evaluate_ndvi_correlation.py

"""
Comprehensive Multi-Sensor NDVI Cross-Validation Pipeline.
Evaluates Landsat (Gap-Filled and Compiled Baseline) against MODIS and AVHRR benchmarks 
strictly INSIDE the AOI Catchment Boundary.
"""

import gc
import re
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

ROOT_DIR = Path(__file__).resolve().parents[1]
GAPFILLED_DIR = ROOT_DIR / "outputs" / "gapfilled"
COMPILED_DIR = ROOT_DIR / "outputs" / "compiled"

MODIS_DIR = ROOT_DIR / "data" / "modis"
AVHRR_DIR = ROOT_DIR / "data" / "avhrr"

SHAPEFILE_PATH = ROOT_DIR / "Delineated_Watershade_final_ExportFeatures.shp"
RESULTS_CSV = ROOT_DIR / "outputs" / "ndvi_correlation_results.csv"

# Global cache for dynamic AOI masks to avoid redundant disk I/O
MASK_CACHE: Dict[Tuple[int, int, str], np.ndarray] = {}


def get_aoi_mask(target_profile: dict) -> np.ndarray:
    """Generates or retrieves a cached boolean mask matching the exact profile grid."""
    cache_key = (
        target_profile["height"],
        target_profile["width"],
        str(target_profile.get("crs")),
    )
    if cache_key in MASK_CACHE:
        return MASK_CACHE[cache_key]

    shp_path = SHAPEFILE_PATH
    if not shp_path.exists():
        found = list(ROOT_DIR.rglob("*.shp"))
        if found:
            shp_path = found[0]
        else:
            raise FileNotFoundError("AOI Catchment Shapefile not found in workspace.")

    aoi_gdf = gpd.read_file(shp_path)

    if aoi_gdf.crs != target_profile["crs"]:
        aoi_gdf = aoi_gdf.to_crs(target_profile["crs"])

    aoi_mask = rasterize(
        shapes=[(geom, 1) for geom in aoi_gdf.geometry],
        out_shape=(target_profile["height"], target_profile["width"]),
        transform=target_profile["transform"],
        fill=0,
        dtype=np.uint8,
    ) == 1

    MASK_CACHE[cache_key] = aoi_mask
    return aoi_mask


def load_or_compute_ndvi(tif_path: Path) -> Tuple[np.ndarray, dict]:
    """Reads single-band NDVI or computes NDVI from 7-band Landsat raster."""
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        if src.count == 1:
            ndvi = src.read(1).astype(np.float32)
        else:
            # Model Config order: [Red (0), Green (1), Blue (2), NIR (3), SWIR1 (4), SWIR2 (5), Thermal (6)]
            red = src.read(1).astype(np.float32)
            nir = src.read(4).astype(np.float32)

            # Auto-scale reflectance if integer 0-10000
            if np.nanmedian(red[red > 0]) > 1.0:
                red = red / 10000.0
            if np.nanmedian(nir[nir > 0]) > 1.0:
                nir = nir / 10000.0

            denom = nir + red
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi = np.where(denom > 1e-4, (nir - red) / denom, np.nan)

        return np.clip(ndvi, -1.0, 1.0), profile


def evaluate_ndvi_pair(
    landsat_path: Path, ref_path: Path, sensor_name: str, scene_type: str
) -> dict:
    """Reprojects reference NDVI to Landsat grid and computes metrics strictly inside AOI."""
    ls_ndvi, ls_profile = load_or_compute_ndvi(landsat_path)
    aoi_mask = get_aoi_mask(ls_profile)

    with rasterio.open(ref_path) as src_ref:
        ref_raw = src_ref.read(1).astype(np.float32)

        # Handle integer scaling (MODIS / AVHRR scaled by 10,000)
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

    # Valid mask: STRICTLY INSIDE AOI + Finite Valid Values
    valid = (
        aoi_mask
        & np.isfinite(ls_ndvi)
        & np.isfinite(ref_resampled)
        & (ls_ndvi >= -1.0) & (ls_ndvi <= 1.0)
        & (ref_resampled >= -1.0) & (ref_resampled <= 1.0)
    )

    if np.sum(valid) < 100:
        return None

    ls_v = ls_ndvi[valid]
    ref_v = ref_resampled[valid]

    std_ls = np.std(ls_v)
    std_ref = np.std(ref_v)

    if std_ls < 1e-6 or std_ref < 1e-6:
        r_val = np.nan
    else:
        r_val, _ = pearsonr(ls_v, ref_v)
        r_val = float(r_val)

    rmse_val = float(np.sqrt(np.mean((ls_v - ref_v) ** 2)))

    ls_2d = np.where(valid, ls_ndvi, np.nanmedian(ls_v))
    ref_2d = np.where(valid, ref_resampled, np.nanmedian(ref_v))
    ssim_val = float(ssim(ls_2d, ref_2d, data_range=2.0))

    match = re.search(r"_(\d{4})_(\d{1,2})", landsat_path.stem)
    year = int(match.group(1)) if match else None
    month = int(match.group(2)) if match else None

    return {
        "year": year,
        "month": month,
        "landsat_file": landsat_path.name,
        "reference_sensor": sensor_name,
        "scene_type": scene_type,
        "pearson_r": r_val,
        "rmse": rmse_val,
        "ssim": ssim_val,
        "valid_pixel_count": int(np.sum(valid)),
    }


def build_reference_map(directory: Path, prefix: str) -> dict:
    """Indexes reference GeoTIFF files by (year, month) tuple."""
    mapping = {}
    if not directory.exists():
        print(f"Warning: Directory does not exist -> {directory}")
        return mapping

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
    print(" Starting AOI-Masked Multi-Sensor NDVI Evaluation Pipeline")
    print(" MODIS Directory: ", MODIS_DIR)
    print(" AVHRR Directory: ", AVHRR_DIR)
    print("=========================================================\n")

    modis_map = build_reference_map(MODIS_DIR, "modis")
    avhrr_map = build_reference_map(AVHRR_DIR, "avhrr")

    print(f"Indexed {len(avhrr_map)} AVHRR reference files (1995_01 to 2022_05).")
    print(f"Indexed {len(modis_map)} MODIS reference files (2000_02 to 2025_12).\n")

    results = []

    # 1. EVALUATE COMPILED DATASET (outputs/compiled/)
    compiled_files = list(COMPILED_DIR.glob("*.tif")) + list(COMPILED_DIR.glob("*.tiff"))
    print(f"Evaluating {len(compiled_files)} compiled scenes in outputs/compiled/...")

    for c_path in compiled_files:
        match = re.search(r"_(\d{4})_(\d{1,2})", c_path.stem)
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))

        scene_type = "Compiled_Raw" if c_path.stem.endswith("_R") else "Compiled_GapFilled"

        if key in avhrr_map:
            res = evaluate_ndvi_pair(c_path, avhrr_map[key], "AVHRR", scene_type)
            if res:
                results.append(res)

        if key in modis_map:
            res = evaluate_ndvi_pair(c_path, modis_map[key], "MODIS", scene_type)
            if res:
                results.append(res)

        gc.collect()

    # SAVE RESULTS
    if results:
        df = pd.DataFrame(results)
        RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
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