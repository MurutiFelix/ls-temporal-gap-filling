# src/compile.py

"""
Compilation Pipeline for Unified Landsat Time-Series (1995–2025).

Assembles a complete dataset into `outputs/compiled/` by:
1. Sourcing 100% complete rasters directly from `outputs/gapfilled/` first,
   retaining their exact `_R.tif` or `_GP.tif` filenames.
2. Sourcing raw scenes from `data/landsat/` ONLY as a fallback if missing in gapfilled,
   applying SLC-off median filtering if needed, and tagging them as `_R.tif`.
3. Enforcing unified Model Config Band Order [Red, Green, Blue, NIR, SWIR1, SWIR2, Thermal].
4. Masking AOI catchment boundaries strictly.
"""

import gc
import re
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy.ndimage import median_filter

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "landsat"
GAPFILLED_DIR = ROOT_DIR / "outputs" / "gapfilled"
COMPILED_DIR = ROOT_DIR / "outputs" / "compiled"

SHAPEFILE_PATH = ROOT_DIR / "Delineated_Watershade_final_ExportFeatures.shp"


def get_aoi_mask(target_profile: dict) -> np.ndarray:
    """Rasterizes the catchment shapefile to create a boolean mask (True inside AOI)."""
    shp_path = SHAPEFILE_PATH
    if not shp_path.exists():
        found = list(ROOT_DIR.rglob("*.shp"))
        if found:
            shp_path = found[0]
        else:
            raise FileNotFoundError("AOI Shapefile not found in workspace.")

    aoi_gdf = gpd.read_file(shp_path)
    aoi_mask = rasterize(
        shapes=[(geom, 1) for geom in aoi_gdf.geometry],
        out_shape=(target_profile["height"], target_profile["width"]),
        transform=target_profile["transform"],
        fill=0,
        dtype=np.uint8,
    ) == 1

    return aoi_mask


def reorder_gee_to_config_bands(data: np.ndarray) -> np.ndarray:
    """
    Reorders bands from GEE export order [Blue, Green, Red, NIR, SWIR1, SWIR2, Thermal]
    to model config order [Red, Green, Blue, NIR, SWIR1, SWIR2, Thermal].
    """
    if data is None or data.ndim != 3 or data.shape[0] < 7:
        return data

    gee_to_config = [2, 1, 0, 3, 4, 5, 6]
    return data[gee_to_config, :, :]


def fill_slc_off_scanlines(raster_cube: np.ndarray) -> np.ndarray:
    """
    Spatially interpolates narrow 1-3 pixel SLC-off scanline gaps in raw Landsat 7 imagery
    using a 3x3 local median filter.
    """
    if raster_cube is None:
        return None

    cleaned = raster_cube.copy()
    for b in range(cleaned.shape[0]):
        band = cleaned[b]
        invalid_mask = (~np.isfinite(band)) | (band <= 0.0001)

        if np.any(invalid_mask):
            filtered = median_filter(band, size=3)
            band[invalid_mask] = filtered[invalid_mask]
            cleaned[b] = band

    return cleaned


def load_and_align_scene(tif_path: Path, target_profile: dict, is_raw: bool = False) -> np.ndarray:
    """
    Loads and reprojects a raster to target_profile.
    Re-orders GEE bands if loading raw imagery.
    """
    with rasterio.open(tif_path) as src:
        out_h = target_profile["height"]
        out_w = target_profile["width"]
        out_crs = target_profile["crs"]
        out_transform = target_profile["transform"]

        destination = np.zeros((src.count, out_h, out_w), dtype=np.float32)

        for band_idx in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=destination[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=out_transform,
                dst_crs=out_crs,
                resampling=Resampling.nearest,
            )

        if np.nanmedian(destination[destination > 0]) > 1.0:
            destination = destination / 10000.0

        destination = np.clip(destination, 0.0, 1.0)
        
        if is_raw:
            destination = reorder_gee_to_config_bands(destination)

        return destination


def find_scene_path(search_dir: Path, year: int, month: int, exclude_pattern: str = "NDVI") -> Path:
    """Locates scene file matching year and month exactly in target folder."""
    if not search_dir.exists():
        return None

    # Fix: Exact month matching using negative lookahead to prevent _1 matching _10, _11, or _12
    pattern = re.compile(rf"_{year}_0?{month}(?!\d)", re.IGNORECASE)

    tif_files = sorted(list(search_dir.glob("*.tif")) + list(search_dir.glob("*.tiff")))
    for f in tif_files:
        if exclude_pattern and exclude_pattern.lower() in f.stem.lower():
            continue
        if pattern.search(f.stem):
            return f
    return None


def get_reference_profile() -> Dict:
    """Extracts reference spatial profile from any available raster."""
    tif_files = list(GAPFILLED_DIR.glob("landsat_*.tif")) or list(RAW_DIR.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError("No reference GeoTIFF files found in gapfilled or raw directories.")

    with rasterio.open(tif_files[0]) as src:
        profile = src.profile.copy()
        profile.update({
            "driver": "GTiff",
            "count": 7,
            "dtype": "float32",
            "nodata": np.nan,
        })
        return profile


def write_compiled_raster(path: Path, array: np.ndarray, profile: Dict) -> None:
    """Writes compiled 7-band GeoTIFF raster to outputs/compiled/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update({
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": "float32",
        "nodata": np.nan,
    })

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(np.where(np.isfinite(array), array, np.nan).astype(np.float32))


def process_month_compilation(
    year: int,
    month: int,
    target_profile: Dict,
    aoi_mask: np.ndarray,
) -> Tuple[str, str, float]:
    """
    Compiles a single target month:
    1. Checks outputs/gapfilled/ first.
    2. Falls back to data/landsat/ if missing in gapfilled.
    """
    gapfilled_path = find_scene_path(GAPFILLED_DIR, year, month, exclude_pattern="NDVI")
    raw_path = find_scene_path(RAW_DIR, year, month, exclude_pattern=None)

    selected_data = None
    compiled_filename = None
    source_type = None

    # 1. PRIORITY 1: GAPFILLED DIRECTORY (Includes both _R and _GP processed scenes)
    if gapfilled_path is not None:
        selected_data = load_and_align_scene(gapfilled_path, target_profile, is_raw=False)
        compiled_filename = gapfilled_path.name
        source_type = "GAPFILLED_" + ("R" if "_R" in gapfilled_path.stem else "GP")

    # 2. PRIORITY 2: RAW LANDSAT FALLBACK (If gapfilled is missing)
    elif raw_path is not None:
        selected_data = load_and_align_scene(raw_path, target_profile, is_raw=True)
        source_type = "RAW_FALLBACK"

        if 2003 <= year <= 2022 and "l7" in raw_path.stem.lower():
            selected_data = fill_slc_off_scanlines(selected_data)

        if "_R" in raw_path.stem:
            compiled_filename = raw_path.name
        else:
            compiled_filename = f"{raw_path.stem}_R.tif"

    if selected_data is None:
        return "MISSING", "", 0.0

    # Mask non-AOI cells
    selected_data[:, ~aoi_mask] = np.nan

    # Calculate AOI coverage
    valid_pixels = np.all(np.isfinite(selected_data) & (selected_data > 0.0001), axis=0) & aoi_mask
    coverage_pct = (np.sum(valid_pixels) / max(np.sum(aoi_mask), 1)) * 100.0

    out_path = COMPILED_DIR / compiled_filename
    write_compiled_raster(out_path, selected_data, target_profile)

    return source_type, compiled_filename, coverage_pct


def main() -> None:
    print("=========================================================")
    print(" Starting Landsat Dataset Compilation (1995 - 2025)...")
    print(" Priority: outputs/gapfilled/ -> data/landsat/")
    print(" Output Folder: outputs/compiled/")
    print("=========================================================\n")

    target_profile = get_reference_profile()
    aoi_mask = get_aoi_mask(target_profile)

    total_compiled = 0
    total_missing = 0

    for year in range(1995, 2026):
        for month in range(1, 13):
            source_type, out_filename, cov_pct = process_month_compilation(
                year=year,
                month=month,
                target_profile=target_profile,
                aoi_mask=aoi_mask,
            )

            if source_type != "MISSING":
                print(f"[{year}-{month:02d}] Compiled -> {out_filename:<32} | Source: {source_type:<15} | AOI Coverage: {cov_pct:.2f}%")
                total_compiled += 1
            else:
                print(f"[{year}-{month:02d}] WARNING: No scene found in gapfilled or raw folders.")
                total_missing += 1

            gc.collect()

    print("\n=========================================================")
    print(" Landsat Compilation Complete!")
    print(f" Successfully compiled months: {total_compiled}")
    print(f" Missing uncompiled months:    {total_missing}")
    print(f" Outputs saved to:             {COMPILED_DIR}")
    print("=========================================================")


if __name__ == "__main__":
    main()