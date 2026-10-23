# src/predict.py

"""
RBFN inference engine for historical Landsat temporal gap filling.
- Retains raw ground-truth pixels in target scenes (< 80% coverage).
- Uses Chained Temporal Search to propagate gap-filled output textures forward.
- Substitutes missing neighbor feature channels with 2D Climatological Composites.
- Includes Automated QA Diagnostics to alert on potential flat blocks in SLURM logs.
"""

import gc
import joblib
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import torch
import yaml
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy.interpolate import griddata
from scipy.ndimage import median_filter

from models.rbfn import RBFN
from preprocessing.raster_processor import RasterProcessor

EXPECTED_IN_FEATURES = 21
EXPECTED_OUT_FEATURES = 7
PREV_AVAILABLE_INDEX = 16
NEXT_AVAILABLE_INDEX = 17
RED_INDEX = 0
NIR_INDEX = 3


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_aoi_mask(target_profile: dict) -> np.ndarray:
    """Rasterizes catchment shapefile to create a boolean mask (True inside AOI)."""
    shp_path = Path("Delineated_Watershade_final_ExportFeatures.shp")
    if not shp_path.exists():
        found = list(Path(".").rglob("*.shp"))
        if found:
            shp_path = found[0]
        else:
            raise FileNotFoundError("AOI Shapefile not found in project workspace.")

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
    """Reorders bands from GEE export order to model config order [Red, Green, Blue, NIR, SWIR1, SWIR2, Thermal]."""
    if data is None or data.ndim != 3 or data.shape[0] < 7:
        return data

    gee_to_config = [2, 1, 0, 3, 4, 5, 6]
    return data[gee_to_config, :, :]


def normalize_reflectance_range(data: np.ndarray) -> np.ndarray:
    """Ensures input imagery is strictly within float reflectance 0.0 - 1.0."""
    if data is None:
        return None

    if np.nanmax(data) > 10.0:
        data = data / 10000.0

    return np.clip(data, 0.0, 1.0)


def fill_slc_off_scanlines(raster_cube: np.ndarray) -> np.ndarray:
    """Spatially interpolates narrow SLC-off scanline gaps in Landsat 7 imagery."""
    if raster_cube is None:
        return None

    cleaned = raster_cube.copy()
    for b in range(cleaned.shape[0]):
        band = cleaned[b]
        invalid_mask = (~np.isfinite(band)) | (band <= 0)

        if np.any(invalid_mask) and np.any(~invalid_mask):
            filtered = median_filter(band, size=3)
            band[invalid_mask] = filtered[invalid_mask]
            cleaned[b] = band

    return cleaned


def find_and_load_target_scene(
    data_dir: Path, year: int, month: int, target_profile: dict
) -> Tuple[np.ndarray, Path]:
    """Scans data_dir for target scene and resamples/reprojects it to target_profile grid."""
    pattern_str_1 = f"_{year}_{month:02d}"
    pattern_str_2 = f"_{year}_{month}"

    tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))

    for tif_path in tif_files:
        stem = tif_path.stem
        if pattern_str_1 in stem or pattern_str_2 in stem:
            with rasterio.open(tif_path) as src:
                out_height = target_profile["height"]
                out_width = target_profile["width"]
                out_crs = target_profile["crs"]
                out_transform = target_profile["transform"]

                destination = np.zeros((src.count, out_height, out_width), dtype=np.float32)

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

                destination = normalize_reflectance_range(destination)
                destination = reorder_gee_to_config_bands(destination)

                return destination, tif_path

    return None, None


def find_chained_temporal_neighbor(
    data_dir: Path,
    output_dir: Path,
    target_year: int,
    target_month: int,
    direction: str,
    target_profile: dict,
    max_search_months: int = 12,
) -> Tuple[np.ndarray, float, int, int]:
    """
    Searches backward/forward up to max_search_months:
    1. Checks output_dir for previously gap-filled outputs (_gapfilled.tif) -> 100% complete texture.
    2. Checks data_dir for raw scenes with >= 50% valid coverage.
    """
    step = -1 if direction == "prev" else 1

    current_yr = target_year
    current_m = target_month

    for offset in range(1, max_search_months + 1):
        current_m += step
        if current_m < 1:
            current_m = 12
            current_yr -= 1
        elif current_m > 12:
            current_m = 1
            current_yr += 1

        gf_path = output_dir / f"landsat_{current_yr}_{current_m:02d}_rbfn_gapfilled.tif"
        if gf_path.exists():
            try:
                with rasterio.open(gf_path) as src:
                    cube = src.read()
                    if cube.shape[0] >= 7 and np.any(np.isfinite(cube) & (cube > 0.0)):
                        return cube[:7, :, :], float(offset), current_yr, current_m
            except Exception:
                pass

        raw_cube, raw_path = find_and_load_target_scene(data_dir, current_yr, current_m, target_profile)
        if raw_cube is not None:
            valid_pixels = np.all(np.isfinite(raw_cube) & (raw_cube > 0.0), axis=0)
            if np.mean(valid_pixels) >= 0.50:
                return raw_cube, float(offset), current_yr, current_m

    fallback_m = (target_month + step * max_search_months - 1) % 12 + 1
    fallback_yr = target_year + (target_month + step * max_search_months - 1) // 12
    return None, float(max_search_months), fallback_yr, fallback_m


def load_climatological_month_composite(
    data_dir: Path, month: int, target_profile: dict
) -> np.ndarray:
    """Computes a multi-year running average composite across historical scenes for target month."""
    tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))
    target_months = [month, (month - 2) % 12 + 1, (month) % 12 + 1]

    out_h = target_profile["height"]
    out_w = target_profile["width"]

    running_sum = np.zeros((7, out_h, out_w), dtype=np.float32)
    running_count = np.zeros((7, out_h, out_w), dtype=np.float32)

    for tif_path in tif_files:
        stem = tif_path.stem.lower()
        matches_month = any(
            f"_{m:02d}_" in stem or f"_{m}_" in stem or stem.endswith(f"_{m:02d}") or stem.endswith(f"_{m}")
            for m in target_months
        )
        if matches_month:
            try:
                with rasterio.open(tif_path) as src:
                    dest = np.zeros((src.count, out_h, out_w), dtype=np.float32)
                    for b in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, b),
                            destination=dest[b - 1],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=target_profile["transform"],
                            dst_crs=target_profile["crs"],
                            resampling=Resampling.nearest,
                        )

                    dest = normalize_reflectance_range(dest)
                    dest = reorder_gee_to_config_bands(dest)

                    valid_mask = np.isfinite(dest) & (dest > 0.005)
                    running_sum[valid_mask] += dest[valid_mask]
                    running_count[valid_mask] += 1.0
            except Exception:
                continue

    valid_composite_mask = running_count > 0
    if not np.any(valid_composite_mask):
        return None

    composite = np.full((7, out_h, out_w), np.nan, dtype=np.float32)
    composite[valid_composite_mask] = (
        running_sum[valid_composite_mask] / running_count[valid_composite_mask]
    )

    for b in range(composite.shape[0]):
        band = composite[b]
        invalid_mask = (~np.isfinite(band)) | (band <= 0.005)
        if np.any(~invalid_mask) and np.any(invalid_mask):
            y_valid, x_valid = np.where(~invalid_mask)
            y_nan, x_nan = np.where(invalid_mask)
            valid_vals = band[~invalid_mask]

            band[invalid_mask] = griddata(
                (y_valid, x_valid), valid_vals, (y_nan, x_nan), method="nearest"
            )
            composite[b] = band

    return composite


def flatten_raster_features(raster_cube: np.ndarray) -> np.ndarray:
    raster_cube = np.asarray(raster_cube, dtype=np.float32)
    bands, height, width = raster_cube.shape
    return np.ascontiguousarray(
        raster_cube.reshape(bands, height * width).T, dtype=np.float32
    )


def validate_scalers(
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = np.asarray(x_mean, dtype=np.float32).ravel()
    x_std = np.asarray(x_std, dtype=np.float32).ravel()
    y_mean = np.asarray(y_mean, dtype=np.float32).ravel()
    y_std = np.asarray(y_std, dtype=np.float32).ravel()

    return x_mean, x_std, y_mean, y_std


def load_model_and_scalers(config: Dict, root_dir: Path):
    model_dir = root_dir / config["paths"]["processed_dir"] / "models"
    model_path = model_dir / "rbfn_landsat_gap_filler.pt"
    scaler_path = model_dir / "rbfn_scalers.joblib"

    rbfn_config = config["model"]["rbfn"]

    model = RBFN(
        in_features=EXPECTED_IN_FEATURES,
        num_centers=int(rbfn_config["num_centers"]),
        out_features=EXPECTED_OUT_FEATURES,
        rbf_chunk_size=int(rbfn_config["chunk_size"]),
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")

    if "centers" in state_dict:
        model.centers = state_dict["centers"].to(model.centers.device)

    model.load_state_dict(state_dict)
    model.eval()

    scalers = joblib.load(scaler_path)

    x_mean, x_std, y_mean, y_std = validate_scalers(
        scalers["x_mean"], scalers["x_std"], scalers["y_mean"], scalers["y_std"]
    )

    x_mean[PREV_AVAILABLE_INDEX] = 0.0
    x_mean[NEXT_AVAILABLE_INDEX] = 0.0
    x_std[PREV_AVAILABLE_INDEX] = 1.0
    x_std[NEXT_AVAILABLE_INDEX] = 1.0

    return model, x_mean, x_std, y_mean, y_std


def predict_in_chunks(model: RBFN, X: np.ndarray, chunk_size: int) -> np.ndarray:
    predictions = []
    with torch.no_grad():
        for start in range(0, X.shape[0], chunk_size):
            end = min(start + chunk_size, X.shape[0])
            X_chunk = torch.from_numpy(np.ascontiguousarray(X[start:end], dtype=np.float32))
            prediction = model(X_chunk)
            predictions.append(prediction.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(predictions, axis=0)


def reconstruct_raster(prediction_physical: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.ascontiguousarray(
        prediction_physical.T.reshape(EXPECTED_OUT_FEATURES, height, width),
        dtype=np.float32,
    )


def derive_ndvi(reconstructed_cube: np.ndarray) -> np.ndarray:
    red = reconstructed_cube[RED_INDEX]
    nir = reconstructed_cube[NIR_INDEX]
    denominator = nir + red

    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(red)
        & np.isfinite(nir)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-8)
    )
    ndvi[valid] = ((nir[valid] - red[valid]) / denominator[valid]).astype(np.float32)
    return ndvi


def run_flat_block_qa_check(
    reconstructed_cube: np.ndarray, aoi_mask: np.ndarray, year: int, month: int
) -> bool:
    """
    Performs spatial variance and unique value diagnostics on output raster.
    Alerts if flat blocks or single value broadcasting is detected.
    """
    red_band = reconstructed_cube[RED_INDEX][aoi_mask]
    valid_red = red_band[np.isfinite(red_band) & (red_band > 0.0)]

    if len(valid_red) == 0:
        print(f"⚠️ [QA ALERT] [{year}-{month:02d}] Zero valid pixels inside AOI boundary!")
        return False

    std_dev = np.std(valid_red)
    unique_count = len(np.unique(np.round(valid_red, decimals=5)))

    if std_dev < 1e-4 or unique_count <= 10:
        print(
            f"⚠️ [QA ALERT] [{year}-{month:02d}] FLAT BLOCK DETECTED! "
            f"StdDev: {std_dev:.6f} | Unique Values: {unique_count:,}"
        )
        return False

    print(
        f"✅ [QA PASS] [{year}-{month:02d}] Spatial Texture Verified. "
        f"StdDev: {std_dev:.4f} | Unique Values: {unique_count:,}"
    )
    return True


def process_gap_month(
    processor: RasterProcessor,
    model: RBFN,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    dem: np.ndarray,
    target_profile: Dict,
    output_dir: Path,
    year: int,
    month: int,
    chunk_size: int,
    output_nodata: float,
    threshold_pct: float = 80.0,
) -> bool:
    aoi_mask = get_aoi_mask(target_profile)
    aoi_mask_flat = aoi_mask.ravel()
    height, width = target_profile["height"], target_profile["width"]
    H, W = height, width

    landsat_dir = Path(__file__).resolve().parents[1] / "data" / "landsat"
    target_cube, scene_path = find_and_load_target_scene(
        landsat_dir, year, month, target_profile
    )

    target_flat = None
    real_obs_mask = np.zeros(aoi_mask_flat.shape, dtype=bool)

    if target_cube is not None:
        target_flat = flatten_raster_features(target_cube)
        real_obs_mask = aoi_mask_flat & np.all(
            np.isfinite(target_flat) & (target_flat > 0.0), axis=1
        )

    total_aoi_pixels = np.sum(aoi_mask_flat)
    valid_aoi_pixels = np.sum(real_obs_mask)
    coverage_pct = (valid_aoi_pixels / max(total_aoi_pixels, 1)) * 100.0

    if coverage_pct >= threshold_pct:
        print(f"[{year}-{month:02d}] Clean Scene ({coverage_pct:.2f}% >= {threshold_pct}%). Skipping.")
        return False

    prev_cube, dt_prev, prev_yr, prev_m = find_chained_temporal_neighbor(
        landsat_dir, output_dir, year, month, "prev", target_profile, max_search_months=12
    )
    next_cube, dt_next, next_yr, next_m = find_chained_temporal_neighbor(
        landsat_dir, output_dir, year, month, "next", target_profile, max_search_months=12
    )

    if prev_cube is None:
        print(f"[{year}-{month:02d}] Prev missing -> Substituting 2D Climatology Composite ({prev_m:02d})")
        prev_cube = load_climatological_month_composite(landsat_dir, prev_m, target_profile)

    if next_cube is None:
        print(f"[{year}-{month:02d}] Next missing -> Substituting 2D Climatology Composite ({next_m:02d})")
        next_cube = load_climatological_month_composite(landsat_dir, next_m, target_profile)

    if prev_cube is None:
        prev_cube = np.repeat(y_mean[:, np.newaxis, np.newaxis], H * W, axis=1).reshape(7, H, W)
    if next_cube is None:
        next_cube = np.repeat(y_mean[:, np.newaxis, np.newaxis], H * W, axis=1).reshape(7, H, W)

    prev_clean = fill_slc_off_scanlines(prev_cube)
    next_clean = fill_slc_off_scanlines(next_cube)

    X = np.zeros((H * W, 21), dtype=np.float32)
    X[:, 0:7] = prev_clean.reshape(7, -1).T
    X[:, 7:14] = next_clean.reshape(7, -1).T
    X[:, 14] = dt_prev
    X[:, 15] = dt_next
    X[:, 16] = 1.0
    X[:, 17] = 1.0
    X[:, 18] = processor.load_era5_predictor("precip", year, month).ravel()
    X[:, 19] = processor.load_era5_predictor("temp", year, month).ravel()
    X[:, 20] = dem.ravel()

    gap_pixels_mask = aoi_mask_flat & (~real_obs_mask)

    prediction_physical = np.full(
        (X.shape[0], EXPECTED_OUT_FEATURES), np.nan, dtype=np.float32
    )

    if valid_aoi_pixels > 0 and target_flat is not None:
        prediction_physical[real_obs_mask] = target_flat[real_obs_mask]

    if np.any(gap_pixels_mask):
        X_valid = X[gap_pixels_mask]
        X_scaled = (X_valid - x_mean) / x_std
        prediction_scaled = predict_in_chunks(model=model, X=X_scaled, chunk_size=chunk_size)
        prediction_physical[gap_pixels_mask] = (prediction_scaled * y_std) + y_mean

    prediction_physical[~aoi_mask_flat] = np.nan

    reconstructed_cube = reconstruct_raster(prediction_physical, height, width)
    ndvi = derive_ndvi(reconstructed_cube)

    # AUTOMATED DIAGNOSTIC CHECK
    run_flat_block_qa_check(reconstructed_cube, aoi_mask, year, month)

    band_output_path = output_dir / f"landsat_{year}_{month:02d}_rbfn_gapfilled.tif"
    ndvi_output_path = output_dir / f"ndvi_{year}_{month:02d}_rbfn_gapfilled.tif"

    write_raster(band_output_path, reconstructed_cube, target_profile, output_nodata)
    write_raster(ndvi_output_path, ndvi, target_profile, output_nodata)

    print(f"[{year}-{month:02d}] Reconstructed Output Saved -> {band_output_path.name}")
    return True


def write_raster(path: Path, array: np.ndarray, profile: Dict, nodata: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    count = 1 if array.ndim == 2 else array.shape[0]

    out_profile.update({
        "driver": "GTiff",
        "height": array.shape[1] if array.ndim == 3 else array.shape[0],
        "width": array.shape[2] if array.ndim == 3 else array.shape[1],
        "count": count,
        "dtype": "float32",
        "nodata": nodata,
    })

    with rasterio.open(path, "w", **out_profile) as dst:
        output = np.where(np.isfinite(array), array, nodata).astype(np.float32)
        if array.ndim == 2:
            dst.write(output, 1)
        else:
            dst.write(output)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    config = load_config(root_dir / "src" / "config.yaml")

    seed = int(config["project"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    model, x_mean, x_std, y_mean, y_std = load_model_and_scalers(config, root_dir)
    processor = RasterProcessor(config)
    dem = processor.load_dem()
    target_profile = processor.get_target_profile()

    gap_start, gap_end = config["landsat"]["gap_years"]
    chunk_size = int(config["model"]["rbfn"]["chunk_size"])
    output_nodata = float(config["gap_filling"]["output_nodata"])
    output_dir = root_dir / config["paths"]["gapfilled_dir"]

    for year in range(int(gap_start), int(gap_end) + 1):
        for month in range(1, 13):
            process_gap_month(
                processor=processor,
                model=model,
                x_mean=x_mean,
                x_std=x_std,
                y_mean=y_mean,
                y_std=y_std,
                dem=dem,
                target_profile=target_profile,
                output_dir=output_dir,
                year=year,
                month=month,
                chunk_size=chunk_size,
                output_nodata=output_nodata,
                threshold_pct=80.0,
            )
            gc.collect()


if __name__ == "__main__":
    main()