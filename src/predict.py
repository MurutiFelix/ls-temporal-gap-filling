# src/predict.py

"""
RBFN inference engine for historical Landsat temporal gap filling.
Reconstructs missing Landsat months using the finalized 21-feature
predictor structure, applies the training-fitted feature and target
scalers, reconstructs all seven Landsat bands, and derives NDVI.

The temporal context is asymmetric at collection boundaries:
- If previous Landsat is available: bands populated, dt_prev set, prev_available = 1
- If previous Landsat is missing: 7 bands zero-filled, dt_prev = 0.0, prev_available = 0
- If next Landsat is available: bands populated, dt_next set, next_available = 1
- If next Landsat is missing: 7 bands zero-filled, dt_next = 0.0, next_available = 0

Processing is skipped only if both prev_available == 0 and next_available == 0.
"""
import joblib
import numpy as np
import rasterio
import torch
import yaml
import gc
import geopandas as gpd
from pathlib import Path
from typing import Dict, Tuple
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from models.rbfn import RBFN
from preprocessing.raster_processor import RasterProcessor

EXPECTED_IN_FEATURES = 21
EXPECTED_OUT_FEATURES = 7
PREVIOUS_BAND_START = 0
PREVIOUS_BAND_END = 7
NEXT_BAND_START = 7
NEXT_BAND_END = 14
DT_PREV_INDEX = 14
DT_NEXT_INDEX = 15
PREV_AVAILABLE_INDEX = 16
NEXT_AVAILABLE_INDEX = 17
ERA5_PRECIP_INDEX = 18
ERA5_TEMP_INDEX = 19
DEM_INDEX = 20
RED_INDEX = 0
NIR_INDEX = 3
BINARY_FEATURE_INDICES = (
    PREV_AVAILABLE_INDEX,
    NEXT_AVAILABLE_INDEX,
)

def find_and_load_target_scene(data_dir: Path, year: int, month: int, target_profile: dict) -> Tuple[np.ndarray, Path]:
    """
    Scans data_dir for the target scene and resamples/reprojects it to match target_profile grid.
    """
    pattern_str_1 = f"_{year}_{month:02d}"
    pattern_str_2 = f"_{year}_{month}"
    
    tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))
    
    for tif_path in tif_files:
        stem = tif_path.stem
        if pattern_str_1 in stem or pattern_str_2 in stem:
            with rasterio.open(tif_path) as src:
                # Target dimensions & transform from target_profile
                out_height = target_profile["height"]
                out_width = target_profile["width"]
                out_crs = target_profile["crs"]
                out_transform = target_profile["transform"]
                
                # Destination array matching exact target profile dimensions
                destination = np.zeros((src.count, out_height, out_width), dtype=np.float32)

                # Reproject / align grid dimensions
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=destination[band_idx - 1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=out_transform,
                        dst_crs=out_crs,
                        resampling=Resampling.nearest
                    )
                
                destination = reorder_gee_to_config_bands(destination)
                
                return destination, tif_path
                
    return None, None

def reorder_gee_to_config_bands(data: np.ndarray) -> np.ndarray:
    """
    Reorders bands from GEE export order [Blue, Green, Red, NIR, SWIR1, SWIR2, Thermal]
    to model config order [Red, Green, Blue, NIR, SWIR1, SWIR2, Thermal].
    """
    if data is None or data.ndim != 3 or data.shape[0] < 7:
        return data

    # GEE indices: 0:Blue, 1:Green, 2:Red, 3:NIR, 4:SWIR1, 5:SWIR2, 6:Thermal
    # Target order: Red (2), Green (1), Blue (0), NIR (3), SWIR1 (4), SWIR2 (5), Thermal (6)
    gee_to_config = [2, 1, 0, 3, 4, 5, 6]
    return data[gee_to_config, :, :]

def get_aoi_mask(target_profile: dict) -> np.ndarray:
    """Rasterizes the catchment shapefile to create a boolean mask (True inside AOI)."""
    shp_path = Path("Delineated_Watershade_final_ExportFeatures.shp")
    if not shp_path.exists():
        found = list(Path(".").rglob("*.shp"))
        if found:
            shp_path = found[0]
        else:
            raise FileNotFoundError("AOI Shapefile not found in project workspace.")

    aoi_gdf = gpd.read_file(shp_path)
    
    # Rasterize shapefile directly onto target profile grid
    aoi_mask = rasterize(
        shapes=[(geom, 1) for geom in aoi_gdf.geometry],
        out_shape=(target_profile["height"], target_profile["width"]),
        transform=target_profile["transform"],
        fill=0,
        dtype=np.uint8,
    ) == 1

    return aoi_mask

def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def fill_slc_off_scanlines(raster_cube: np.ndarray) -> np.ndarray:
    """
    Spatially interpolates narrow 1-3 pixel SLC-off scanline gaps in Landsat 7 imagery
    using a 3x3 local median filter before passing as RBFN features.
    """
    if raster_cube is None:
        return None
        
    cleaned = raster_cube.copy()
    for b in range(cleaned.shape[0]):
        band = cleaned[b]
        invalid_mask = (~np.isfinite(band)) | (band <= 0)
        
        if np.any(invalid_mask):
            filtered = median_filter(band, size=3)
            # Patch only missing scanline pixels with local spatial median
            band[invalid_mask] = filtered[invalid_mask]
            cleaned[b] = band
            
    return cleaned

def flatten_raster_features(
    raster_cube: np.ndarray,
) -> np.ndarray:
    raster_cube = np.asarray(
        raster_cube,
        dtype=np.float32,
    )
    if raster_cube.ndim != 3:
        raise ValueError(
            "Expected raster cube with shape "
            f"(bands, height, width), got {raster_cube.shape}."
        )

    if raster_cube.shape[0] != EXPECTED_OUT_FEATURES:
        raise ValueError(
            "Expected a seven-band Landsat cube, "
            f"got {raster_cube.shape[0]} bands."
        )

    bands, height, width = raster_cube.shape

    return np.ascontiguousarray(
        raster_cube.reshape(
            bands,
            height * width,
        ).T,
        dtype=np.float32,
    )


def validate_scalers(
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = np.asarray(x_mean).ravel()
    x_std = np.asarray(x_std).ravel()
    y_mean = np.asarray(y_mean).ravel()
    y_std = np.asarray(y_std).ravel()

    expected_x_shape = (EXPECTED_IN_FEATURES,)
    expected_y_shape = (EXPECTED_OUT_FEATURES,)

    if x_mean.shape != expected_x_shape:
        raise ValueError(
            f"x_mean must have shape {expected_x_shape}, got {x_mean.shape}."
        )

    if x_std.shape != expected_x_shape:
        raise ValueError(
            f"x_std must have shape {expected_x_shape}, got {x_std.shape}."
        )

    if y_mean.shape != expected_y_shape:
        raise ValueError(
            f"y_mean must have shape {expected_y_shape}, got {y_mean.shape}."
        )

    if y_std.shape != expected_y_shape:
        raise ValueError(
            f"y_std must have shape {expected_y_shape}, got {y_std.shape}."
        )

    arrays = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    for name, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values.")

    if np.any(x_std <= 0):
        raise ValueError("x_std contains zero or negative values.")

    if np.any(y_std <= 0):
        raise ValueError("y_std contains zero or negative values.")

    return x_mean, x_std, y_mean, y_std


def load_model_and_scalers(
    config: Dict,
    root_dir: Path,
) -> Tuple[
    RBFN,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model_dir = (
        root_dir
        / config["paths"]["processed_dir"]
        / "models"
    )
    model_path = (
        model_dir
        / "rbfn_landsat_gap_filler.pt"
    )

    scaler_path = (
        model_dir
        / "rbfn_scalers.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler checkpoint not found: {scaler_path}"
        )

    rbfn_config = config["model"]["rbfn"]

    model = RBFN(
        in_features=EXPECTED_IN_FEATURES,
        num_centers=int(
            rbfn_config["num_centers"]
        ),
        out_features=EXPECTED_OUT_FEATURES,
        rbf_chunk_size=int(
            rbfn_config["chunk_size"]
        ),
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(
            "The model checkpoint does not contain 'state_dict' or 'model_state_dict'."
        )
 
    if "centers" in state_dict:
        model.centers = state_dict["centers"].to(model.centers.device)

    model.load_state_dict(state_dict)

    model.eval()

    scalers = joblib.load(
        scaler_path
    )

    if not isinstance(scalers, dict):
        raise TypeError(
            "The scaler checkpoint must contain a dictionary."
        )

    required_keys = {
        "x_mean",
        "x_std",
        "y_mean",
        "y_std",
    }

    missing_keys = required_keys.difference(
        scalers.keys()
    )

    if missing_keys:
        raise KeyError(
            "Scaler checkpoint is missing required keys: "
            f"{sorted(missing_keys)}."
        )

    x_mean = np.asarray(
        scalers["x_mean"],
        dtype=np.float32,
    )

    x_std = np.asarray(
        scalers["x_std"],
        dtype=np.float32,
    )

    y_mean = np.asarray(
        scalers["y_mean"],
        dtype=np.float32,
    )

    y_std = np.asarray(
        scalers["y_std"],
        dtype=np.float32,
    )


    x_mean, x_std, y_mean, y_std = validate_scalers(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    binary_feature_indices = scalers.get(
        "binary_feature_indices",
        list(BINARY_FEATURE_INDICES),
    )

    if tuple(binary_feature_indices) != BINARY_FEATURE_INDICES:
        raise ValueError(
            "Unexpected binary feature indices in scaler "
            f"checkpoint: {binary_feature_indices}."
        )

    # Make explicit copies of the sanitized 1D arrays
    x_mean = x_mean.copy()
    x_std = x_std.copy()

    # Zero-out shift and unit-scale availability flag indices
    x_mean[PREV_AVAILABLE_INDEX] = 0.0
    x_mean[NEXT_AVAILABLE_INDEX] = 0.0

    x_std[PREV_AVAILABLE_INDEX] = 1.0
    x_std[NEXT_AVAILABLE_INDEX] = 1.0

    return (
        model,
        x_mean,
        x_std,
        y_mean,
        y_std,
    )


def validate_temporal_context(
    neighbors: Dict,
    year: int,
    month: int,
) -> Tuple[int, int]:
    """Validates the structure of the temporal neighbors dictionary.

    Returns the integer availability flags (prev_available,
    next_available).
    """
    required_keys = {
        "landsat_prev",
        "landsat_next",
        "dt_prev",
        "dt_next",
        "prev_available",
        "next_available",
    }
    missing_keys = required_keys.difference(
        neighbors.keys()
    )

    if missing_keys:
        raise KeyError(
            "Temporal-neighbour result for "
            f"{year}-{month:02d} is missing keys: "
            f"{sorted(missing_keys)}."
        )

    prev_available = int(
        neighbors["prev_available"]
    )

    next_available = int(
        neighbors["next_available"]
    )

    if prev_available not in (0, 1):
        raise ValueError(
            f"Invalid prev_available for "
            f"{year}-{month:02d}: "
            f"{neighbors['prev_available']}."
        )

    if next_available not in (0, 1):
        raise ValueError(
            f"Invalid next_available for "
            f"{year}-{month:02d}: "
            f"{neighbors['next_available']}."
        )

    if prev_available == 1:
        if neighbors["landsat_prev"] is None:
            raise ValueError(
                f"prev_available is 1 but no previous Landsat "
                f"observation was supplied for {year}-{month:02d}."
            )

        if not np.isfinite(
            float(neighbors["dt_prev"])
        ):
            raise ValueError(
                f"Invalid dt_prev for {year}-{month:02d}."
            )

    if next_available == 1:
        if neighbors["landsat_next"] is None:
            raise ValueError(
                f"next_available is 1 but no next Landsat "
                f"observation was supplied for {year}-{month:02d}."
            )

        if not np.isfinite(
            float(neighbors["dt_next"])
        ):
            raise ValueError(
                f"Invalid dt_next for {year}-{month:02d}."
            )

    if (
        prev_available == 0
        and next_available == 0
    ):
        raise ValueError(
            f"No temporal Landsat context is available within "
            f"the configured search window for {year}-{month:02d}."
        )

    return prev_available, next_available

def build_inference_features(processor, year, month, dem, aoi_mask):
    H, W = dem.shape
    spatial_shape = (H, W)

    neighbors = processor.load_temporal_neighbors(year, month)
    prev_cube = fill_slc_off_scanlines(reorder_gee_to_config_bands(neighbors.get("landsat_prev")))
    next_cube = fill_slc_off_scanlines(reorder_gee_to_config_bands(neighbors.get("landsat_next")))

    # Relaxed per-pixel availability: True where at least one band is valid and > 0
    has_prev = (
        np.any(np.isfinite(prev_cube) & (prev_cube > 0), axis=0)
        if prev_cube is not None
        else np.zeros(spatial_shape, dtype=bool)
    )
    has_next = (
        np.any(np.isfinite(next_cube) & (next_cube > 0), axis=0)
        if next_cube is not None
        else np.zeros(spatial_shape, dtype=bool)
    )

    prev_clean = np.nan_to_num(prev_cube, nan=0.0) if prev_cube is not None else np.zeros((7, H, W), dtype=np.float32)
    next_clean = np.nan_to_num(next_cube, nan=0.0) if next_cube is not None else np.zeros((7, H, W), dtype=np.float32)
    
    X = np.zeros((H * W, 21), dtype=np.float32)
    X[:, 0:7]   = prev_clean.reshape(7, -1).T
    X[:, 7:14]  = next_clean.reshape(7, -1).T
    X[:, 14]    = float(neighbors.get("dt_prev", 12.0))
    X[:, 15]    = float(neighbors.get("dt_next", 12.0))
    X[:, 16]    = has_prev.ravel().astype(np.float32)
    X[:, 17]    = has_next.ravel().astype(np.float32)
    
    X[:, 18]    = processor.load_era5_predictor("precip", year, month).ravel()
    X[:, 19]    = processor.load_era5_predictor("temp", year, month).ravel()
    X[:, 20]    = dem.ravel()

    # Valid mask requires pixels to be strictly INSIDE the AOI and have valid environmental predictors
    aoi_mask_flat = aoi_mask.ravel()
    valid_mask = aoi_mask_flat & np.all(np.isfinite(X[:, 18:21]), axis=1)

    return X, valid_mask, has_prev, has_next


def predict_in_chunks(
    model: RBFN,
    X: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature matrix, got {X.shape}."
        )
    if X.shape[1] != EXPECTED_IN_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_IN_FEATURES} input features, "
            f"got {X.shape[1]}."
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if X.shape[0] == 0:
        return np.empty(
            (
                0,
                EXPECTED_OUT_FEATURES,
            ),
            dtype=np.float32,
        )

    predictions = []

    with torch.no_grad():
        for start in range(
            0,
            X.shape[0],
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                X.shape[0],
            )

            X_chunk = torch.from_numpy(
                np.ascontiguousarray(
                    X[start:end],
                    dtype=np.float32,
                )
            )

            prediction = model(
                X_chunk
            )

            predictions.append(
                prediction.cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            )

    return np.concatenate(
        predictions,
        axis=0,
    )


def reconstruct_raster(
    prediction_physical: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    if prediction_physical.ndim != 2:
        raise ValueError(
            "Prediction array must be 2D."
        )
    if prediction_physical.shape[1] != EXPECTED_OUT_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_OUT_FEATURES} output bands, "
            f"got {prediction_physical.shape[1]}."
        )

    expected_pixels = height * width

    if prediction_physical.shape[0] != expected_pixels:
        raise ValueError(
            "Prediction pixel count does not match "
            "the target grid."
        )

    return np.ascontiguousarray(
        prediction_physical.T.reshape(
            EXPECTED_OUT_FEATURES,
            height,
            width,
        ),
        dtype=np.float32,
    )


def derive_ndvi(
    reconstructed_cube: np.ndarray,
) -> np.ndarray:
    if reconstructed_cube.ndim != 3:
        raise ValueError(
            "Reconstructed Landsat cube must have shape "
            "(bands, height, width)."
        )
    if reconstructed_cube.shape[0] != EXPECTED_OUT_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_OUT_FEATURES} reconstructed bands, "
            f"got {reconstructed_cube.shape[0]}."
        )

    red = reconstructed_cube[
        RED_INDEX
    ]

    nir = reconstructed_cube[
        NIR_INDEX
    ]

    denominator = nir + red

    ndvi = np.full(
        red.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(red)
        & np.isfinite(nir)
        & np.isfinite(denominator)
        & (
            np.abs(denominator)
            > 1e-8
        )
    )

    ndvi[valid] = (
        (
            nir[valid]
            - red[valid]
        )
        / denominator[valid]
    ).astype(
        np.float32
    )

    return ndvi

def normalize_reflectance_range(data: np.ndarray) -> np.ndarray:
    """Ensures input imagery is strictly within float reflectance 0.0 - 1.0."""
    if data is None:
        return None
        
    # If imagery was saved in 0 - 10000 surface reflectance scale (e.g. integer SR)
    if np.nanmax(data) > 10.0:
        data = data / 10000.0
        
    return np.clip(data, 0.0, 1.0)

def blend_boundary_seamless(real_obs_mask_2d, target_cube, prediction_cube):
    """
    Feathers the junction between real observations and RBFN predictions
    using a 5-pixel spatial blend to eliminate orbital swath seams.
    """
    # Create smooth float weight mask (1.0 = Ground Truth, 0.0 = RBFN Prediction)
    weight_mask = real_obs_mask_2d.astype(np.float32)
    smoothed_weights = gaussian_filter(weight_mask, sigma=2.0)
    
    # Expand 2D weight to all 7 bands
    smoothed_weights_3d = np.repeat(smoothed_weights[np.newaxis, :, :], 7, axis=0)
    
    # Blend target and prediction seamlessly
    blended = (target_cube * smoothed_weights_3d) + (prediction_cube * (1.0 - smoothed_weights_3d))
    return blended

def write_raster(
    path: Path,
    array: np.ndarray,
    profile: Dict,
    nodata: float,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_profile = profile.copy()

    if array.ndim == 2:
        height, width = array.shape
        count = 1

    elif array.ndim == 3:
        count, height, width = array.shape

    else:
        raise ValueError(
            f"Raster array must be 2D or 3D, "
            f"got {array.shape}."
        )

    if (
        height != output_profile["height"]
        or width != output_profile["width"]
    ):
        raise ValueError(
            "Raster dimensions do not match "
            "the target profile."
        )

    output_profile.update(
        {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": count,
            "dtype": "float32",
            "nodata": nodata,
        }
    )

    with rasterio.open(
        path,
        "w",
        **output_profile,
    ) as dst:
        output = np.where(
            np.isfinite(array),
            array,
            nodata,
        ).astype(
            np.float32,
            copy=False,
        )

        if output.ndim == 2:
            dst.write(
                output,
                1,
            )
        else:
            dst.write(
                output
            )

def load_climatological_month_composite(
    data_dir: Path, 
    month: int, 
    target_profile: dict
) -> np.ndarray:
    """
    Computes a memory-efficient running mean composite across historical scenes
    to avoid Slurm OOM kills on large spatial grids.
    """
    tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))
    target_months = [month, (month - 2) % 12 + 1, (month) % 12 + 1]
    
    out_h = target_profile["height"]
    out_w = target_profile["width"]

    # Running sum and count matrices to keep RAM usage low (Shape: 7, H, W)
    running_sum = np.zeros((7, out_h, out_w), dtype=np.float32)
    running_count = np.zeros((7, out_h, out_w), dtype=np.float32)

    for tif_path in tif_files:
        stem = tif_path.stem
        if any(f"_{m:02d}_" in stem or f"_{m}_" in stem for m in target_months):
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
                    dest = reorder_gee_to_config_bands(dest)
                    
                    valid_mask = np.isfinite(dest) & (dest > 0)
                    running_sum[valid_mask] += dest[valid_mask]
                    running_count[valid_mask] += 1.0
                    
                    del dest
            except Exception:
                continue

    # Compute mean composite where at least 1 valid observation exists
    valid_composite_mask = running_count > 0
    if not np.any(valid_composite_mask):
        return None

    composite = np.full((7, out_h, out_w), np.nan, dtype=np.float32)
    composite[valid_composite_mask] = running_sum[valid_composite_mask] / running_count[valid_composite_mask]

    # Free up intermediate matrices
    del running_sum, running_count
    gc.collect()

    # Spatial inpainting for remaining NaNs
    for b in range(composite.shape[0]):
        band = composite[b]
        nan_mask = np.isnan(band)
        if np.any(nan_mask) and not np.all(nan_mask):
            y_valid, x_valid = np.where(~nan_mask)
            y_nan, x_nan = np.where(nan_mask)
            valid_vals = band[~nan_mask]
            
            band[nan_mask] = griddata(
                (y_valid, x_valid), 
                valid_vals, 
                (y_nan, x_nan), 
                method="nearest"
            )
            composite[b] = band

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
    # 1. Build spatial AOI boundary mask
    aoi_mask = get_aoi_mask(target_profile)
    aoi_mask_flat = aoi_mask.ravel()

    # 2. DIRECT DISK LOAD: Locate target scene in data/landsat
    landsat_dir = Path(__file__).resolve().parents[1] / "data" / "landsat"
    target_cube, scene_path = find_and_load_target_scene(landsat_dir, year, month, target_profile)
    
    target_flat = None
    real_obs_mask = np.zeros(aoi_mask_flat.shape, dtype=bool)

    if target_cube is not None:
        target_flat = flatten_raster_features(target_cube)
        # Identify ground truth pixels strictly INSIDE AOI that have valid reflectance (>0 and finite)
        real_obs_mask = aoi_mask_flat & np.all(np.isfinite(target_flat) & (target_flat > 0), axis=1)
        print(f"[{year}-{month:02d}] Loaded raw scene from disk: {scene_path.name}")
    else:
        print(f"[{year}-{month:02d}] No raw scene file found on disk. Treating as 0% coverage (Full Gap).")

    # Calculate actual AOI coverage percentage
    total_aoi_pixels = np.sum(aoi_mask_flat)
    valid_aoi_pixels = np.sum(real_obs_mask)
    coverage_pct = (valid_aoi_pixels / max(total_aoi_pixels, 1)) * 100.0

    print(f"[{year}-{month:02d}] Calculated AOI Coverage: {coverage_pct:.2f}%")

   
    if coverage_pct >= threshold_pct:
        print(f"[{year}-{month:02d}] AOI Coverage ({coverage_pct:.2f}%) >= Threshold ({threshold_pct}%). Skipping gap-filling.")
        return False

    # 3. Build features for gap filling
    try:
        X_raw, valid_mask, has_prev, has_next = build_inference_features(
            processor=processor,
            year=year,
            month=month,
            dem=dem,
            aoi_mask=aoi_mask,
        )
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"Skipping {year}-{month:02d}: {exc}")
        return False

    # 4. Identify missing gap pixels within AOI boundary
    gap_pixels_mask = valid_mask & (~real_obs_mask)

    has_prev_flat = has_prev.ravel()
    has_next_flat = has_next.ravel()

    pred_mask = gap_pixels_mask & (has_prev_flat | has_next_flat)
    fallback_mask = gap_pixels_mask & ~(has_prev_flat | has_next_flat)

    print(f"[{year}-{month:02d}] Retained Ground Truth Pixels: {valid_aoi_pixels:,}")
    print(f"[{year}-{month:02d}] Gap Pixels RBFN Filled: {np.sum(pred_mask):,}")
    print(f"[{year}-{month:02d}] Isolated Gap Pixels Fallback: {np.sum(fallback_mask):,}")

    prediction_physical = np.full(
        (X_raw.shape[0], EXPECTED_OUT_FEATURES),
        np.nan,
        dtype=np.float32,
    )

    # A. PRESERVE REAL OBSERVATIONS FIRST
    if valid_aoi_pixels > 0 and target_flat is not None:
        prediction_physical[real_obs_mask] = target_flat[real_obs_mask]

    # B. RBFN PREDICTION FOR GAP PIXELS ONLY
    if np.any(pred_mask):
        X_valid = X_raw[pred_mask]
        X_scaled = (X_valid - x_mean) / x_std
        X_scaled = np.ascontiguousarray(X_scaled, dtype=np.float32)

        prediction_scaled = predict_in_chunks(
            model=model,
            X=X_scaled,
            chunk_size=chunk_size,
        )

        prediction_physical[pred_mask] = (prediction_scaled * y_std) + y_mean

    # C. FALLBACK FOR ISOLATED GAP PIXELS
    num_fallback = np.sum(fallback_mask)
    if num_fallback > 0:
        # FAST PATH: Small residual gap (< 100,000 pixels) -> borrow nearest temporal neighbor or y_mean
        if num_fallback < 100000:
            print(f"[{year}-{month:02d}] Fast local fallback for {num_fallback:,} remaining edge pixels...")
            prev_flat = X_raw[:, 0:7]
            next_flat = X_raw[:, 7:14]
            neighbor_avg = np.where(prev_flat > 0, prev_flat, next_flat)
            fallback_vals = np.where(neighbor_avg > 0, neighbor_avg, y_mean)
            prediction_physical[fallback_mask] = fallback_vals[fallback_mask]

        # HEAVY GAP PATH (>= 100,000 pixels) -> trigger climatological search across disk
        else:
            print(f"[{year}-{month:02d}] Heavy gap ({num_fallback:,} pixels missing). Synthesizing via climatological historical composite...")
            clim_composite = load_climatological_month_composite(
                data_dir=landsat_dir, 
                month=month, 
                target_profile=target_profile
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

    # Mask out non-AOI background cells
    prediction_physical[~aoi_mask_flat] = np.nan

    # Reconstruct 3D GeoTIFF raster cube
    height = int(target_profile["height"])
    width = int(target_profile["width"])

    reconstructed_cube = reconstruct_raster(
        prediction_physical=prediction_physical,
        height=height,
        width=width,
    )

    ndvi = derive_ndvi(reconstructed_cube)

    band_output_path = output_dir / f"landsat_{year}_{month:02d}_rbfn_gapfilled.tif"
    ndvi_output_path = output_dir / f"ndvi_{year}_{month:02d}_rbfn_gapfilled.tif"

    write_raster(
        path=band_output_path,
        array=reconstructed_cube,
        profile=target_profile,
        nodata=output_nodata,
    )

    write_raster(
        path=ndvi_output_path,
        array=ndvi,
        profile=target_profile,
        nodata=output_nodata,
    )

    print(f"Saved bands: {band_output_path}")
    print(f"Saved NDVI: {ndvi_output_path}")

    return True

def main() -> None:
    root_dir = (
        Path(__file__).resolve().parents[1]
    )
    config_path = (
        root_dir
        / "src"
        / "config.yaml"
    )

    config = load_config(
        config_path
    )

    seed = int(
        config["project"]["seed"]
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    (
        model,
        x_mean,
        x_std,
        y_mean,
        y_std,
    ) = load_model_and_scalers(
        config=config,
        root_dir=root_dir,
    )

    processor = RasterProcessor(
        config
    )

    dem = processor.load_dem()

    target_profile = (
        processor.get_target_profile()
    )

    gap_start, gap_end = config[
        "landsat"
    ][
        "gap_years"
    ]

    chunk_size = int(
        config[
            "model"
        ][
            "rbfn"
        ][
            "chunk_size"
        ]
    )

    output_nodata = float(
        config[
            "gap_filling"
        ][
            "output_nodata"
        ]
    )

    output_dir = (
        root_dir
        / config[
            "paths"
        ][
            "gapfilled_dir"
        ]
    )

    total_processed = 0
    total_skipped = 0
    total_observed = 0

    for year in range(
        int(gap_start),
        int(gap_end) + 1,
    ):
        for month in range(1, 13):
            print(f"Processing gap month: {year}-{month:02d}")

            processed = process_gap_month(
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

            if processed:
                total_processed += 1
            else:
                total_skipped += 1

            gc.collect()

    print(
        "Gap-filling inference completed."
    )

    print(
        f"Observed clean months skipped: "
        f"{total_observed}"
    )

    print(
        f"Gap months reconstructed: "
        f"{total_processed}"
    )

    print(
        f"Gap months skipped: "
        f"{total_skipped}"
    )

if __name__ == "__main__":
    main()