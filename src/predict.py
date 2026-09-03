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
from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import rasterio
import torch
import yaml
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


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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
) -> None:
    expected_x_shape = (
        EXPECTED_IN_FEATURES,
    )
    expected_y_shape = (
        EXPECTED_OUT_FEATURES,
    )

    if x_mean.shape != expected_x_shape:
        raise ValueError(
            f"x_mean must have shape {expected_x_shape}, "
            f"got {x_mean.shape}."
        )

    if x_std.shape != expected_x_shape:
        raise ValueError(
            f"x_std must have shape {expected_x_shape}, "
            f"got {x_std.shape}."
        )

    if y_mean.shape != expected_y_shape:
        raise ValueError(
            f"y_mean must have shape {expected_y_shape}, "
            f"got {y_mean.shape}."
        )

    if y_std.shape != expected_y_shape:
        raise ValueError(
            f"y_std must have shape {expected_y_shape}, "
            f"got {y_std.shape}."
        )

    arrays = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    for name, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} contains non-finite values."
            )

    if np.any(x_std <= 0):
        raise ValueError(
            "x_std contains zero or negative values."
        )

    if np.any(y_std <= 0):
        raise ValueError(
            "y_std contains zero or negative values."
        )


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

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The model checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

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

    validate_scalers(
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

    x_mean = x_mean.copy()
    x_std = x_std.copy()

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


def build_inference_features(
    processor: RasterProcessor,
    year: int,
    month: int,
    dem: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    neighbors = processor.load_temporal_neighbors(
        year,
        month,
    )

    prev_available, next_available = validate_temporal_context(
        neighbors=neighbors,
        year=year,
        month=month,
    )

    dem = np.asarray(
        dem,
        dtype=np.float32,
    )

    if dem.ndim != 2:
        raise ValueError(
            "DEM must be a 2D raster. "
            f"Got {dem.shape}."
        )

    height, width = dem.shape
    num_pixels = height * width

    X = np.zeros(
        (
            num_pixels,
            EXPECTED_IN_FEATURES,
        ),
        dtype=np.float32,
    )

    # 1. Previous temporal context
    if prev_available == 1:
        landsat_prev = np.asarray(
            neighbors["landsat_prev"],
            dtype=np.float32,
        )

        previous_matrix = flatten_raster_features(
            landsat_prev
        )

        if previous_matrix.shape[0] != num_pixels:
            raise ValueError(
                "Previous Landsat observation does not match "
                "the target grid."
            )

        X[
            :,
            PREVIOUS_BAND_START:PREVIOUS_BAND_END,
        ] = previous_matrix

        X[:, DT_PREV_INDEX] = np.float32(
            neighbors["dt_prev"]
        )
    else:
        X[
            :,
            PREVIOUS_BAND_START:PREVIOUS_BAND_END,
        ] = 0.0

        X[:, DT_PREV_INDEX] = 0.0

    # 2. Next temporal context
    if next_available == 1:
        landsat_next = np.asarray(
            neighbors["landsat_next"],
            dtype=np.float32,
        )

        next_matrix = flatten_raster_features(
            landsat_next
        )

        if next_matrix.shape[0] != num_pixels:
            raise ValueError(
                "Next Landsat observation does not match "
                "the target grid."
            )

        X[
            :,
            NEXT_BAND_START:NEXT_BAND_END,
        ] = next_matrix

        X[:, DT_NEXT_INDEX] = np.float32(
            neighbors["dt_next"]
        )
    else:
        X[
            :,
            NEXT_BAND_START:NEXT_BAND_END,
        ] = 0.0

        X[:, DT_NEXT_INDEX] = 0.0

    # 3. Availability flags
    X[:, PREV_AVAILABLE_INDEX] = np.float32(
        prev_available
    )

    X[:, NEXT_AVAILABLE_INDEX] = np.float32(
        next_available
    )

    # 4. Environmental & spatial predictors
    precip = np.asarray(
        processor.load_era5_predictor("precip"),
        dtype=np.float32,
    )

    temp = np.asarray(
        processor.load_era5_predictor("temp"),
        dtype=np.float32,
    )

    if precip.ndim != 2:
        raise ValueError(
            "ERA5 precipitation must be a 2D raster. "
            f"Got {precip.shape}."
        )

    if temp.ndim != 2:
        raise ValueError(
            "ERA5 temperature must be a 2D raster. "
            f"Got {temp.shape}."
        )

    if precip.shape != dem.shape:
        raise ValueError(
            "ERA5 precipitation does not match the target grid. "
            f"Expected {dem.shape}, got {precip.shape}."
        )

    if temp.shape != dem.shape:
        raise ValueError(
            "ERA5 temperature does not match the target grid. "
            f"Expected {dem.shape}, got {temp.shape}."
        )

    X[:, ERA5_PRECIP_INDEX] = precip.reshape(
        -1
    )

    X[:, ERA5_TEMP_INDEX] = temp.reshape(
        -1
    )

    X[:, DEM_INDEX] = dem.reshape(
        -1
    )

    # Filter out non-finite pixels (e.g. background/nodata pixels in DEM or ERA5)
    valid_mask = np.all(
        np.isfinite(X),
        axis=1,
    )

    valid_mask &= np.isin(
        X[:, PREV_AVAILABLE_INDEX],
        [0.0, 1.0],
    )

    valid_mask &= np.isin(
        X[:, NEXT_AVAILABLE_INDEX],
        [0.0, 1.0],
    )

    return (
        np.ascontiguousarray(
            X,
            dtype=np.float32,
        ),
        valid_mask,
    )


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
) -> bool:
    try:
        X_raw, valid_mask = build_inference_features(
            processor=processor,
            year=year,
            month=month,
            dem=dem,
        )
    except (
        KeyError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        print(
            f"Skipping {year}-{month:02d}: {exc}"
        )

        return False

    prediction_scaled = np.full(
        (
            X_raw.shape[0],
            EXPECTED_OUT_FEATURES,
        ),
        np.nan,
        dtype=np.float32,
    )

    if not np.any(valid_mask):
        print(
            f"Skipping {year}-{month:02d}: "
            "no valid inference pixels."
        )

        return False

    X_valid = X_raw[
        valid_mask
    ]

    X_scaled = (
        X_valid
        - x_mean
    ) / x_std

    X_scaled = np.ascontiguousarray(
        X_scaled,
        dtype=np.float32,
    )

    prediction_scaled[
        valid_mask
    ] = predict_in_chunks(
        model=model,
        X=X_scaled,
        chunk_size=chunk_size,
    )

    prediction_physical = np.full(
        prediction_scaled.shape,
        np.nan,
        dtype=np.float32,
    )

    prediction_physical[
        valid_mask
    ] = (
        prediction_scaled[
            valid_mask
        ] * y_std
    ) + y_mean

    height = int(
        target_profile["height"]
    )

    width = int(
        target_profile["width"]
    )

    reconstructed_cube = reconstruct_raster(
        prediction_physical=prediction_physical,
        height=height,
        width=width,
    )

    ndvi = derive_ndvi(
        reconstructed_cube
    )

    band_output_path = (
        output_dir
        / f"landsat_{year}_{month:02d}_rbfn_gapfilled.tif"
    )

    ndvi_output_path = (
        output_dir
        / f"ndvi_{year}_{month:02d}_rbfn_gapfilled.tif"
    )

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

    print(
        f"Saved bands: {band_output_path}"
    )

    print(
        f"Saved NDVI: {ndvi_output_path}"
    )

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

    available_months = set(
        processor.get_available_months()
    )

    total_processed = 0
    total_skipped = 0
    total_observed = 0

    for year in range(
        int(gap_start),
        int(gap_end) + 1,
    ):
        for month in range(
            1,
            13,
        ):
            if (
                year,
                month,
            ) in available_months:
                total_observed += 1

                print(
                    f"Observed Landsat month, "
                    f"skipping: "
                    f"{year}-{month:02d}"
                )

                continue

            print(
                f"Processing gap month: "
                f"{year}-{month:02d}"
            )

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

    print(
        "Gap-filling inference completed."
    )

    print(
        f"Observed months skipped: "
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