# src/predict.py

"""
RBFN inference engine for historical Landsat temporal gap filling.
Guarantees 100% valid pixel coverage for all generated gap months via a 3-tier 
reconstruction hierarchy: RBFN -> Climatological Composite -> Spatial Catchment Mean.
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
from scipy.ndimage import median_filter

from models.rbfn import RBFN
from preprocessing.raster_processor import RasterProcessor

EXPECTED_IN_FEATURES = 21
EXPECTED_OUT_FEATURES = 7
PREV_AVAILABLE_INDEX = 16
NEXT_AVAILABLE_INDEX = 17
RED_INDEX = 0
NIR_INDEX = 3
BINARY_FEATURE_INDICES = (PREV_AVAILABLE_INDEX, NEXT_AVAILABLE_INDEX)


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

                destination = reorder_gee_to_config_bands(destination)

                return destination, tif_path

    return None, None


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
    """Flattens 2D joblib scaler arrays into strict 1D vectors."""
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


def build_inference_features(processor, year, month, dem, aoi_mask):
    H, W = dem.shape
    spatial_shape = (H, W)

    neighbors = processor.load_temporal_neighbors(year, month)
    prev_cube = fill_slc_off_scanlines(neighbors.get("landsat_prev"))
    next_cube = fill_slc_off_scanlines(neighbors.get("landsat_next"))

    aoi_total_pixels = np.sum(aoi_mask)

    prev_valid_pixels = (
        np.all(np.isfinite(prev_cube) & (prev_cube > 0.001), axis=0) & aoi_mask
        if prev_cube is not None
        else np.zeros(spatial_shape, dtype=bool)
    )
    next_valid_pixels = (
        np.all(np.isfinite(next_cube) & (next_cube > 0.001), axis=0) & aoi_mask
        if next_cube is not None
        else np.zeros(spatial_shape, dtype=bool)
    )

    prev_cov_pct = (np.sum(prev_valid_pixels) / max(aoi_total_pixels, 1)) * 100.0
    next_cov_pct = (np.sum(next_valid_pixels) / max(aoi_total_pixels, 1)) * 100.0

    use_prev_for_rbfn = prev_cov_pct >= 80.0
    use_next_for_rbfn = next_cov_pct >= 80.0

    has_prev = prev_valid_pixels if use_prev_for_rbfn else np.zeros(spatial_shape, dtype=bool)
    has_next = next_valid_pixels if use_next_for_rbfn else np.zeros(spatial_shape, dtype=bool)

    prev_clean = (
        np.nan_to_num(prev_cube, nan=0.0)
        if prev_cube is not None
        else np.zeros((7, H, W), dtype=np.float32)
    )
    next_clean = (
        np.nan_to_num(next_cube, nan=0.0)
        if next_cube is not None
        else np.zeros((7, H, W), dtype=np.float32)
    )

    X = np.zeros((H * W, 21), dtype=np.float32)
    X[:, 0:7] = prev_clean.reshape(7, -1).T
    X[:, 7:14] = next_clean.reshape(7, -1).T
    X[:, 14] = float(neighbors.get("dt_prev", 12.0))
    X[:, 15] = float(neighbors.get("dt_next", 12.0))
    X[:, 16] = has_prev.ravel().astype(np.float32)
    X[:, 17] = has_next.ravel().astype(np.float32)

    X[:, 18] = processor.load_era5_predictor("precip", year, month).ravel()
    X[:, 19] = processor.load_era5_predictor("temp", year, month).ravel()
    X[:, 20] = dem.ravel()

    aoi_mask_flat = aoi_mask.ravel()
    valid_mask = aoi_mask_flat & np.all(np.isfinite(X[:, 18:21]), axis=1)

    return X, valid_mask, has_prev, has_next


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


def load_climatological_month_composite(
    data_dir: Path, month: int, target_profile: dict
) -> np.ndarray:
    """Computes a multi-year running average composite across all historical Landsat scenes."""
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

                    if np.nanmax(dest) > 10.0:
                        dest = dest / 10000.0
                    dest = np.clip(dest, 0.0, 1.0)
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

    return composite


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
        print(f"[{year}-{month:02d}] Raw Scene Coverage Found: {scene_path.name}")

    total_aoi_pixels = np.sum(aoi_mask_flat)
    valid_aoi_pixels = np.sum(real_obs_mask)
    coverage_pct = (valid_aoi_pixels / max(total_aoi_pixels, 1)) * 100.0

    # Rule: Skip non-gap months (scenes with >= 80% clean raw coverage)
    if coverage_pct >= threshold_pct:
        print(f"[{year}-{month:02d}] Clean Month ({coverage_pct:.2f}% raw coverage >= {threshold_pct}%). Skipping gap filling.")
        return False

    try:
        X_raw, valid_mask, has_prev, has_next = build_inference_features(
            processor=processor, year=year, month=month, dem=dem, aoi_mask=aoi_mask
        )
    except Exception as exc:
        print(f"Skipping {year}-{month:02d}: {exc}")
        return False

    # Gap pixels needing reconstruction
    gap_pixels_mask = aoi_mask_flat & (~real_obs_mask)
    has_prev_flat = has_prev.ravel()
    has_next_flat = has_next.ravel()

    pred_mask = gap_pixels_mask & (has_prev_flat | has_next_flat)
    fallback_mask = gap_pixels_mask & ~(has_prev_flat | has_next_flat)

    prediction_physical = np.full(
        (X_raw.shape[0], EXPECTED_OUT_FEATURES), np.nan, dtype=np.float32
    )

    # 1. PRESERVE GROUND TRUTH
    if valid_aoi_pixels > 0 and target_flat is not None:
        prediction_physical[real_obs_mask] = target_flat[real_obs_mask]

    # 2. TIER 1: RBFN NEURAL NETWORK GAP FILL
    if np.any(pred_mask):
        X_valid = X_raw[pred_mask]
        X_scaled = (X_valid - x_mean) / x_std
        prediction_scaled = predict_in_chunks(model=model, X=X_scaled, chunk_size=chunk_size)
        prediction_physical[pred_mask] = (prediction_scaled * y_std) + y_mean

    # 3. TIER 2: CLIMATOLOGICAL HISTORICAL COMPOSITE FALLBACK
    num_fallback = np.sum(fallback_mask)
    if num_fallback > 0:
        clim_composite = load_climatological_month_composite(
            data_dir=landsat_dir, month=month, target_profile=target_profile
        )
        if clim_composite is not None:
            clim_flat = flatten_raster_features(clim_composite)
            prediction_physical[fallback_mask] = clim_flat[fallback_mask]
        else:
            prev_flat = X_raw[:, 0:7]
            next_flat = X_raw[:, 7:14]
            neighbor_avg = np.where(prev_flat > 0, prev_flat, next_flat)
            fallback_vals = np.where(neighbor_avg > 0, neighbor_avg, y_mean)
            prediction_physical[fallback_mask] = fallback_vals[fallback_mask]

    # 4. TIER 3: GUARANTEE 100% CATCHMENT COVERAGE (SPATIAL MEAN CATCHMENT FILL)
    unassigned_aoi = aoi_mask_flat & (~np.all(np.isfinite(prediction_physical) & (prediction_physical > 0.0), axis=1))
    if np.any(unassigned_aoi):
        # Assign fitted monthly global mean vector to any remaining unassigned catchment cells
        prediction_physical[unassigned_aoi] = y_mean

    # Mask out background non-AOI cells outside the catchment boundary
    prediction_physical[~aoi_mask_flat] = np.nan

    reconstructed_cube = reconstruct_raster(prediction_physical, height, width)
    ndvi = derive_ndvi(reconstructed_cube)

    band_output_path = output_dir / f"landsat_{year}_{month:02d}_rbfn_gapfilled.tif"
    ndvi_output_path = output_dir / f"ndvi_{year}_{month:02d}_rbfn_gapfilled.tif"

    write_raster(band_output_path, reconstructed_cube, target_profile, output_nodata)
    write_raster(ndvi_output_path, ndvi, target_profile, output_nodata)

    valid_out_pixels = np.sum(np.all(np.isfinite(reconstructed_cube) & (reconstructed_cube > 0), axis=0) & aoi_mask)
    out_coverage_pct = (valid_out_pixels / max(total_aoi_pixels, 1)) * 100.0

    print(f"[{year}-{month:02d}] Reconstructed Output Catchment Coverage: {out_coverage_pct:.2f}% | Saved -> {band_output_path.name}")
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
            )
            gc.collect()


if __name__ == "__main__":
    main()