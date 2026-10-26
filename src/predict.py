# src/predict.py

"""
RBFN inference engine for historical Landsat temporal gap filling (v4).

Goal: every month in the gap period leaves as a complete, seamless 7-band
Landsat image (plus NDVI) with real image texture and no constant-value blocks.

Fill hierarchy (processing is strictly chronological):

  1. OBSERVED   real Landsat pixels of the target month are kept untouched.
  2. RBFN       every gap pixel is predicted by the RBFN. Its neighbours are,
                per pixel and in this order of preference:
                  a. the nearest REAL previous / next observation inside the
                     search window,
                  b. otherwise the PREVIOUS MONTH'S GAP-FILLED image (the
                     chain), with dt set to the pixel's age (months since its
                     last real observation, capped at the window).
                Because the chain is always available after the first month,
                the "both neighbours missing" case no longer occurs.
  3. GUARD      An RBF network returns (almost) its bias when an input is far
                from every training centre, which is the same value for the
                whole area: a flat block. Each RBFN prediction is therefore
                compared with a REFERENCE image that carries real texture:
                  reference = multi-year climatology of the same calendar
                              month (real 30 m pixels from other years),
                              else the previous filled month, else the nearest
                              real neighbour.
                  * Predictions may differ from the reference by at most
                    MAX_ANOMALY_SIGMAS training standard deviations.
                  * Where the prediction has lost its texture (local std far
                    below the reference's, or constant), it is discarded and
                    replaced by  reference + smooth regional offset. The offset
                    is learned from this month's real pixels (observed minus
                    climatology) and from the healthy RBFN pixels, so the
                    image keeps real texture and matches the month's state.
             The guard also catches OUT-OF-DISTRIBUTION pixels exactly: the model
             is  readout(exp(-gamma * d^2))  over the K-means centres, so a pixel
             far from every centre outputs only the ridge intercept (one value
             for the whole area). Such pixels are detected from the centre
             distances and never written.
             dt units (months or days) and the longest dt seen in training are
             read from the training scaler, so inputs stay inside the range the
             model was fitted on.
  4. SPATIAL    only pixels never observed in any year and with no reference
                are filled from the nearest valid pixel and smoothed.

Usage
-----
  python src/predict.py                    all gap months in config
  python src/predict.py 1996-01 1997-06    only this range (inclusive), still
                                           chronological, so the chain works.
                                           Use it to check a heavy-gap stretch
                                           before running everything.

Outputs per month
-----------------
  landsat_YYYY_MM_rbfn_gapfilled.tif      7 bands, config order
  ndvi_YYYY_MM_rbfn_gapfilled.tif         NDVI
  provenance/landsat_YYYY_MM_source.tif   uint8: 1 observed, 2 RBFN,
                                          3 RBFN guarded (flat or out-of-distribution),
                                          4 climatology, 5 previous filled
                                          month / nearest neighbour, 6 spatial
  preview/landsat_YYYY_MM.png             quick-look RGB (needs Pillow)
  gapfill_run_summary.csv                 coverage, source shares and a
                                          flat-block QA figure per month
"""
import csv
import gc
import re
import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import joblib
import numpy as np
import rasterio
import torch
import yaml
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt, gaussian_filter, uniform_filter, zoom

from models.rbfn import RBFN
from preprocessing.raster_processor import RasterProcessor

EXPECTED_IN_FEATURES = 21
EXPECTED_OUT_FEATURES = 7
N_BANDS = 7
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

# ---------------------------------------------------------------- tunables --
# Months searched either side of the target for a REAL observation. Can be
# overridden with config["gap_filling"]["search_window_months"].
SEARCH_WINDOW_MONTHS = 12
# Units of the dt features. None = infer from the training scaler (a mean dt
# above 20 means days). Set 1.0 (months) or 30.4 (days) to force.
DT_PER_MONTH_OVERRIDE = None
# Longest dt (months) fed to the model. None = infer from the training scaler
# (mean + 2 std of the training dt), never above the search window.
MAX_DT_MONTHS_OVERRIDE = None
# A pixel is out-of-distribution when it is further than OOD_FACTOR x (95th
# percentile nearest-neighbour spacing of the RBF centres) from every centre.
OOD_FACTOR = 2.0
# True: also fill the residual gaps of months that already have >= 80 % real
# coverage, so every month of the series is complete.
PROCESS_SUFFICIENT_MONTHS = True
# True: the previous month's gap-filled image is used as a neighbour wherever no
# real previous observation exists inside the window.
USE_FILLED_AS_NEIGHBOUR = True
# Number of full scenes kept in memory (each is 7 x H x W float32).
SCENE_CACHE_SIZE = 3
# Output values are clipped to y_mean +/- CLIP_SIGMAS * y_std (lower bound >= 0).
CLIP_SIGMAS = 8.0
# Smoothing (pixels) applied only to spatially filled residual pixels.
SPATIAL_SMOOTH_SIGMA = 3.0

# --- texture guard ---------------------------------------------------------
# Maximum departure of an RBFN prediction from the reference image, in training
# standard deviations of the target bands.
MAX_ANOMALY_SIGMAS = 1.5
# Window (pixels) for the local-texture test and bands it is applied to.
TEXTURE_WINDOW = 31
TEXTURE_BANDS = (RED_INDEX, NIR_INDEX)
# A prediction is "flat" if its local std is below this share of the reference's
# local std, or below FLAT_ABS_SIGMAS training std (i.e. constant).
FLAT_RATIO = 0.35
FLAT_ABS_SIGMAS = 0.01
# Smoothing scale (pixels) of the regional offset applied in flat areas, and the
# block factor used to compute it cheaply.
OFFSET_SIGMA_PX = 60.0
OFFSET_DOWNSAMPLE = 8
# Quick-look PNGs.
WRITE_PREVIEWS = True
PREVIEW_STEP = 8

# Provenance codes
SRC_NONE = 0
SRC_OBSERVED = 1
SRC_RBFN = 2
SRC_GUARDED = 3
SRC_CLIMATOLOGY = 4
SRC_CHAIN = 5
SRC_SPATIAL = 6

_SCENE_RE = re.compile(r"_(\d{4})_(\d{1,2})(?!\d)")


# ------------------------------------------------------------------ config --
def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ------------------------------------------------------------ band helpers --
def reorder_gee_to_config_bands(data: np.ndarray) -> np.ndarray:
    """
    Reorders bands from GEE export order [Blue, Green, Red, NIR, SWIR1, SWIR2, Thermal]
    to model config order [Red, Green, Blue, NIR, SWIR1, SWIR2, Thermal].
    """
    if data is None or data.ndim != 3 or data.shape[0] < 7:
        return data
    gee_to_config = [2, 1, 0, 3, 4, 5, 6]
    return data[gee_to_config, :, :]


def band_limits(y_mean: np.ndarray, y_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Physically plausible per-band range derived from the training target scaler."""
    lo = np.maximum(y_mean - CLIP_SIGMAS * y_std, 0.0).astype(np.float32)
    hi = (y_mean + CLIP_SIGMAS * y_std).astype(np.float32)
    return lo, hi


def shift_month(year: int, month: int, delta: int) -> Tuple[int, int]:
    t = year * 12 + (month - 1) + delta
    return t // 12, t % 12 + 1


# -------------------------------------------------------------- AOI + scenes --
def get_aoi_mask(target_profile: dict) -> np.ndarray:
    """Rasterizes the catchment shapefile to a boolean mask (True inside AOI)."""
    shp_path = Path("Delineated_Watershade_final_ExportFeatures.shp")
    if not shp_path.exists():
        found = list(Path(".").rglob("*.shp"))
        if found:
            shp_path = found[0]
        else:
            raise FileNotFoundError("AOI Shapefile not found in project workspace.")

    aoi_gdf = gpd.read_file(shp_path)
    if aoi_gdf.crs is not None and target_profile.get("crs") is not None:
        if aoi_gdf.crs != target_profile["crs"]:
            aoi_gdf = aoi_gdf.to_crs(target_profile["crs"])

    aoi_mask = rasterize(
        shapes=[(geom, 1) for geom in aoi_gdf.geometry],
        out_shape=(target_profile["height"], target_profile["width"]),
        transform=target_profile["transform"],
        fill=0,
        dtype=np.uint8,
    ) == 1
    return aoi_mask


def index_scenes(data_dir: Path) -> Dict[Tuple[int, int], Path]:
    """Maps (year, month) -> real Landsat GeoTIFF found in data_dir."""
    index: Dict[Tuple[int, int], Path] = {}
    files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))
    for path in sorted(files):
        match = _SCENE_RE.search(path.stem)
        if not match:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            index.setdefault((year, month), path)
    return index


class SceneStore:
    """
    Loads real Landsat scenes onto the target grid as (7, H, W) float32 cubes in
    model band order. Invalid pixels are NaN. Scale (0-10000 vs 0-1) is inferred
    per band from the training target scaler, so every code path is consistent.
    """

    def __init__(
        self,
        index: Dict[Tuple[int, int], Path],
        target_profile: dict,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        cache_size: int = SCENE_CACHE_SIZE,
    ) -> None:
        self.index = index
        self.profile = target_profile
        self.y_mean = np.asarray(y_mean, dtype=np.float32)
        self.lo, self.hi = band_limits(self.y_mean, np.asarray(y_std, dtype=np.float32))
        self.cache_size = cache_size
        self._cache: "OrderedDict[Tuple[int, int], Optional[np.ndarray]]" = OrderedDict()

    def _same_grid(self, src) -> bool:
        p = self.profile
        if src.height != p["height"] or src.width != p["width"]:
            return False
        if src.crs != p["crs"]:
            return False
        return bool(
            np.allclose(
                tuple(src.transform)[:6], tuple(p["transform"])[:6], rtol=1e-12, atol=1e-9
            )
        )

    def _harmonise_scale(self, data: np.ndarray) -> np.ndarray:
        step = max(1, int(np.sqrt(data.shape[1] * data.shape[2] / 50000.0)))
        sample = data[:, ::step, ::step]
        for b in range(N_BANDS):
            vals = sample[b]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size < 20 or self.y_mean[b] <= 0:
                continue
            if float(np.median(vals)) / float(self.y_mean[b]) > 50.0:
                data[b] /= 10000.0
        return data

    def read(self, path: Path) -> Optional[np.ndarray]:
        """Reads one scene (uncached)."""
        p = self.profile
        h, w = int(p["height"]), int(p["width"])
        try:
            with rasterio.open(path) as src:
                if src.count < N_BANDS:
                    print(f"  ! {path.name}: only {src.count} bands, skipped.")
                    return None
                nodata = src.nodata
                if self._same_grid(src):
                    data = src.read(indexes=list(range(1, N_BANDS + 1)), out_dtype="float32")
                    if nodata is not None and np.isfinite(nodata):
                        data[data == nodata] = np.nan
                else:
                    data = np.zeros((N_BANDS, h, w), dtype=np.float32)
                    for b in range(1, N_BANDS + 1):
                        reproject(
                            source=rasterio.band(src, b),
                            destination=data[b - 1],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            src_nodata=nodata,
                            dst_transform=p["transform"],
                            dst_crs=p["crs"],
                            dst_nodata=0.0,
                            resampling=Resampling.nearest,
                        )
        except Exception as exc:
            print(f"  ! Could not read {path.name}: {exc}")
            return None

        data = reorder_gee_to_config_bands(data)
        data = np.ascontiguousarray(data, dtype=np.float32)
        data = self._harmonise_scale(data)
        data[~(data > 0)] = np.nan  # NaN, zero and negative values are invalid
        np.clip(data, self.lo[:, None, None], self.hi[:, None, None], out=data)
        return data

    def get(self, year: int, month: int) -> Optional[np.ndarray]:
        key = (year, month)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self.index.get(key)
        if path is None:
            return None
        cube = self.read(path)
        self._cache[key] = cube
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cube


# ------------------------------------------------------- model + scalers --
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
        raise ValueError(f"x_mean must have shape {expected_x_shape}, got {x_mean.shape}.")
    if x_std.shape != expected_x_shape:
        raise ValueError(f"x_std must have shape {expected_x_shape}, got {x_std.shape}.")
    if y_mean.shape != expected_y_shape:
        raise ValueError(f"y_mean must have shape {expected_y_shape}, got {y_mean.shape}.")
    if y_std.shape != expected_y_shape:
        raise ValueError(f"y_std must have shape {expected_y_shape}, got {y_std.shape}.")

    arrays = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
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
) -> Tuple[RBFN, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_dir = root_dir / config["paths"]["processed_dir"] / "models"
    model_path = model_dir / "rbfn_landsat_gap_filler.pt"
    scaler_path = model_dir / "rbfn_scalers.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler checkpoint not found: {scaler_path}")

    rbfn_config = config["model"]["rbfn"]

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    num_centers = int(checkpoint.get("num_centers", rbfn_config["num_centers"]))
    if num_centers != int(rbfn_config["num_centers"]):
        print(
            f"Note: checkpoint has {num_centers} RBF centres, config says "
            f"{rbfn_config['num_centers']}. Using the checkpoint value."
        )

    model = RBFN(
        in_features=EXPECTED_IN_FEATURES,
        num_centers=num_centers,
        out_features=EXPECTED_OUT_FEATURES,
        rbf_chunk_size=int(rbfn_config["chunk_size"]),
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

    scalers = joblib.load(scaler_path)
    if not isinstance(scalers, dict):
        raise TypeError("The scaler checkpoint must contain a dictionary.")

    required_keys = {"x_mean", "x_std", "y_mean", "y_std"}
    missing_keys = required_keys.difference(scalers.keys())
    if missing_keys:
        raise KeyError(
            f"Scaler checkpoint is missing required keys: {sorted(missing_keys)}."
        )

    x_mean = np.asarray(scalers["x_mean"], dtype=np.float32)
    x_std = np.asarray(scalers["x_std"], dtype=np.float32)
    y_mean = np.asarray(scalers["y_mean"], dtype=np.float32)
    y_std = np.asarray(scalers["y_std"], dtype=np.float32)

    x_mean, x_std, y_mean, y_std = validate_scalers(
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std
    )

    binary_feature_indices = scalers.get(
        "binary_feature_indices", list(BINARY_FEATURE_INDICES)
    )
    if tuple(binary_feature_indices) != BINARY_FEATURE_INDICES:
        raise ValueError(
            f"Unexpected binary feature indices in scaler checkpoint: {binary_feature_indices}."
        )

    x_mean = x_mean.copy()
    x_std = x_std.copy()

    # Availability flags are binary: no shift, unit scale.
    x_mean[PREV_AVAILABLE_INDEX] = 0.0
    x_mean[NEXT_AVAILABLE_INDEX] = 0.0
    x_std[PREV_AVAILABLE_INDEX] = 1.0
    x_std[NEXT_AVAILABLE_INDEX] = 1.0

    return model, x_mean, x_std, y_mean, y_std


# ------------------------------------------------------ predictor handling --
def sanitize_predictor(array: np.ndarray, fallback: float, label: str) -> np.ndarray:
    """
    Returns a flat float32 predictor with no NaN / inf. Missing cells take the
    median of the valid cells, or the training mean if nothing is valid. A NaN
    predictor must never remove a pixel from the output.
    """
    a = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(a)
    if finite.all():
        return np.ascontiguousarray(a.ravel())
    fill = float(np.median(a[finite])) if finite.any() else float(fallback)
    pct = 100.0 * (1.0 - finite.mean())
    print(f"  ! {label}: {pct:.1f}% non-finite cells replaced with {fill:.4f}")
    return np.ascontiguousarray(np.where(finite, a, fill).astype(np.float32).ravel())


def infer_dt_settings(
    x_mean: np.ndarray, x_std: np.ndarray, window: int
) -> Tuple[float, int]:
    """
    Reads the dt units and the longest training dt from the training scaler.
    Returns (dt_per_month, max_dt_months). max_dt_months only limits the dt VALUE
    given to the model; the search for real neighbours still spans `window`.
    """
    idx = (DT_PREV_INDEX, DT_NEXT_INDEX)
    dt_mean = float(max(x_mean[i] for i in idx))
    per_month = (
        float(DT_PER_MONTH_OVERRIDE)
        if DT_PER_MONTH_OVERRIDE
        else (30.4 if dt_mean > 20.0 else 1.0)
    )
    hi = float(max(x_mean[i] + 2.0 * x_std[i] for i in idx))
    cap = (
        int(MAX_DT_MONTHS_OVERRIDE)
        if MAX_DT_MONTHS_OVERRIDE
        else max(1, int(round(hi / per_month)))
    )
    max_dt = int(max(1, cap))
    print(
        f"dt features: training mean {dt_mean:.2f} -> {'days' if per_month > 1.5 else 'months'} "
        f"(x{per_month:g} per month); dt values are clamped to {max_dt} months "
        f"(training range); real-neighbour search window stays {window} months."
    )
    return per_month, max_dt


@torch.no_grad()
def rbf_ood_distance(model: RBFN) -> float:
    """Distance to the nearest RBF centre beyond which a pixel is out-of-distribution."""
    c = model.centers.detach().float()
    d = torch.cdist(c, c)
    d.fill_diagonal_(float("inf"))
    nn_spacing = d.min(dim=1).values
    return float(OOD_FACTOR * torch.quantile(nn_spacing, 0.95))


# ------------------------------------------------- per-pixel temporal context --
def collect_temporal_context(
    store: SceneStore,
    year: int,
    month: int,
    gap_idx: np.ndarray,
    window: int,
    chain: Optional[np.ndarray] = None,
    age: Optional[np.ndarray] = None,
    dt_per_month: float = 1.0,
    max_dt: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    For every gap pixel, finds the nearest REAL previous and next observation
    within `window` months. Where no real previous observation exists, the
    previous month's gap-filled image (`chain`) is used instead, with dt equal to
    the pixel's age in months (capped at `window`). Values, dt and availability
    are all per pixel; where nothing is found, bands stay zero, dt stays 0 and
    the flag stays 0, exactly the convention used in training.
    """
    n = gap_idx.size
    dt_cap = int(max_dt) if max_dt is not None else int(window)
    ctx = {
        "prev_vals": np.zeros((n, N_BANDS), dtype=np.float32),
        "next_vals": np.zeros((n, N_BANDS), dtype=np.float32),
        "prev_dt": np.zeros(n, dtype=np.float32),
        "next_dt": np.zeros(n, dtype=np.float32),
        "prev_ok": np.zeros(n, dtype=bool),
        "next_ok": np.zeros(n, dtype=bool),
        "prev_filled": np.zeros(n, dtype=bool),
    }

    for k in range(1, window + 1):
        for direction, delta in (("prev", -k), ("next", k)):
            ok = ctx[f"{direction}_ok"]
            if ok.all():
                continue
            y, m = shift_month(year, month, delta)
            scene = store.get(y, m)
            if scene is None:
                continue
            todo = np.flatnonzero(~ok)
            vals = scene.reshape(N_BANDS, -1)[:, gap_idx[todo]].T
            good = np.all(np.isfinite(vals), axis=1)
            sel = todo[good]
            if sel.size:
                ctx[f"{direction}_vals"][sel] = vals[good]
                ctx[f"{direction}_dt"][sel] = min(k, dt_cap) * dt_per_month
                ok[sel] = True
        if ctx["prev_ok"].all() and ctx["next_ok"].all():
            break

    # Chain: previous month's gap-filled image as the previous neighbour.
    if USE_FILLED_AS_NEIGHBOUR and chain is not None:
        need = np.flatnonzero(~ctx["prev_ok"])
        if need.size:
            vals = chain.reshape(N_BANDS, -1)[:, gap_idx[need]].T
            good = np.all(np.isfinite(vals), axis=1)
            sel = need[good]
            if sel.size:
                ages = age[gap_idx[sel]] if age is not None else window
                ctx["prev_vals"][sel] = vals[good]
                ctx["prev_dt"][sel] = np.minimum(ages + 1, dt_cap) * dt_per_month
                ctx["prev_ok"][sel] = True
                ctx["prev_filled"][sel] = True
    return ctx


def predict_gap_pixels(
    model: RBFN,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    gap_idx: np.ndarray,
    sel: np.ndarray,
    ctx: Dict[str, np.ndarray],
    era5_precip: np.ndarray,
    era5_temp: np.ndarray,
    dem_flat: np.ndarray,
    chunk_size: int,
    ood_dist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """RBFN inference for gap_idx[sel], built and scaled chunk by chunk.
    Returns (predictions, out_of_distribution flags)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    n = sel.size
    out = np.empty((n, EXPECTED_OUT_FEATURES), dtype=np.float32)
    ood = np.zeros(n, dtype=bool)

    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            rows = sel[start:end]
            px = gap_idx[rows]

            X = np.zeros((end - start, EXPECTED_IN_FEATURES), dtype=np.float32)
            X[:, PREVIOUS_BAND_START:PREVIOUS_BAND_END] = ctx["prev_vals"][rows]
            X[:, NEXT_BAND_START:NEXT_BAND_END] = ctx["next_vals"][rows]
            X[:, DT_PREV_INDEX] = ctx["prev_dt"][rows]
            X[:, DT_NEXT_INDEX] = ctx["next_dt"][rows]
            X[:, PREV_AVAILABLE_INDEX] = ctx["prev_ok"][rows]
            X[:, NEXT_AVAILABLE_INDEX] = ctx["next_ok"][rows]
            X[:, ERA5_PRECIP_INDEX] = era5_precip[px]
            X[:, ERA5_TEMP_INDEX] = era5_temp[px]
            X[:, DEM_INDEX] = dem_flat[px]

            X_scaled = np.ascontiguousarray((X - x_mean) / x_std, dtype=np.float32)
            phi = model.get_rbf_features(torch.from_numpy(X_scaled))
            pred_scaled = model.readout(phi).cpu().numpy()
            a_max = phi.max(dim=1).values.clamp_min(1e-30)
            d_min = torch.sqrt(-torch.log(a_max) / model.gamma.clamp_min(1e-12))
            ood[start:end] = (d_min > ood_dist).cpu().numpy()
            out[start:end] = pred_scaled * y_std + y_mean

    return out, ood


# ----------------------------------------------------------- climatology --
def _accumulate_composite(
    store: SceneStore, scene_keys: list, h: int, w: int
) -> np.ndarray:
    """Per-pixel mean of the given real scenes (NaN where no scene is valid)."""
    total = np.zeros((N_BANDS, h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for key in scene_keys:
        cube = store.read(store.index[key])
        if cube is None:
            continue
        ok = np.all(np.isfinite(cube), axis=0)
        total[:, ok] += cube[:, ok]
        count[ok] += 1.0
        del cube, ok
    composite = np.full((N_BANDS, h, w), np.nan, dtype=np.float32)
    has = count > 0
    composite[:, has] = total[:, has] / count[has]
    del total, count
    gc.collect()
    return composite


def build_climatology(
    store: SceneStore, month: int, aoi_mask: np.ndarray
) -> Optional[np.ndarray]:
    """
    Multi-year composite of one calendar month at full resolution. Exact-month
    scenes are used first; adjacent months only fill pixels the exact month
    never observed, so seasonality is not blurred where real data exists.
    """
    h, w = aoi_mask.shape
    exact, adjacent = [], []
    for (y, m) in store.index:
        diff = min((m - month) % 12, (month - m) % 12)
        if diff == 0:
            exact.append((y, m))
        elif diff == 1:
            adjacent.append((y, m))

    if not exact and not adjacent:
        return None

    composite = (
        _accumulate_composite(store, exact, h, w)
        if exact
        else np.full((N_BANDS, h, w), np.nan, dtype=np.float32)
    )
    missing = aoi_mask & ~np.isfinite(composite[0])
    if adjacent and missing.any():
        adj = _accumulate_composite(store, adjacent, h, w)
        use = missing & np.isfinite(adj[0])
        composite[:, use] = adj[:, use]
        del adj
        gc.collect()
    return composite


class ClimatologyProvider:
    """Builds each calendar-month composite once, caches it on disk, and keeps
    the most recent one in memory. Delete the cache folder to rebuild."""

    def __init__(self, store: SceneStore, aoi_mask: np.ndarray, cache_dir: Path) -> None:
        self.store = store
        self.aoi_mask = aoi_mask
        self.cache_dir = cache_dir
        self._month: Optional[int] = None
        self._data: Optional[np.ndarray] = None

    def get(self, month: int) -> Optional[np.ndarray]:
        if self._month == month:
            return self._data

        self._data = None
        gc.collect()
        path = self.cache_dir / f"climatology_{month:02d}.tif"
        data = None
        if path.exists():
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)
        else:
            print(f"  Building climatology for calendar month {month:02d} (one-off)...")
            data = build_climatology(self.store, month, self.aoi_mask)
            if data is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                profile = self.store.profile.copy()
                profile.update(
                    driver="GTiff",
                    count=N_BANDS,
                    dtype="float32",
                    nodata=np.nan,
                    compress="deflate",
                    BIGTIFF="IF_SAFER",
                )
                with rasterio.open(path, "w", **profile) as dst:
                    dst.write(data)

        self._month, self._data = month, data
        return data


# ----------------------------------------------------------- texture guard --
def local_std(band: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    """Local standard deviation of `band` over `size` x `size` windows, using only
    pixels where `mask` is True. NaN where fewer than 30 % of the window is valid."""
    m = mask.astype(np.float32)
    a = np.where(mask, band, 0.0).astype(np.float32)
    n = uniform_filter(m, size, mode="constant")
    s1 = uniform_filter(a, size, mode="constant")
    s2 = uniform_filter(a * a, size, mode="constant")
    safe = np.maximum(n, 1e-6)
    mean = s1 / safe
    var = np.maximum(s2 / safe - mean * mean, 0.0)
    sd = np.sqrt(var).astype(np.float32)
    sd[n < 0.3] = np.nan
    return sd


def detect_flat(
    pred_c: np.ndarray,
    ref_c: np.ndarray,
    gap_idx: np.ndarray,
    rb: np.ndarray,
    shape: Tuple[int, int],
    y_std: np.ndarray,
) -> np.ndarray:
    """
    Returns a bool array (len gap_idx) marking RBFN pixels whose prediction has
    lost its image texture: local std far below the reference's, or constant.
    """
    h, w = shape
    n_total = h * w
    rbm = np.zeros(n_total, dtype=bool)
    rbm[gap_idx[rb]] = True
    rbm2d = rbm.reshape(h, w)

    flat2d = np.zeros((h, w), dtype=bool)
    for b in TEXTURE_BANDS:
        pg = np.full(n_total, np.nan, dtype=np.float32)
        pg[gap_idx] = pred_c[:, b]
        rg = np.full(n_total, np.nan, dtype=np.float32)
        rg[gap_idx] = ref_c[:, b]
        pg, rg = pg.reshape(h, w), rg.reshape(h, w)

        ps = local_std(pg, rbm2d & np.isfinite(pg), TEXTURE_WINDOW)
        rs = local_std(rg, rbm2d & np.isfinite(rg), TEXTURE_WINDOW)
        floor = FLAT_ABS_SIGMAS * float(y_std[b])
        is_flat = np.isfinite(ps) & (
            (ps < floor) | (np.isfinite(rs) & (rs > floor) & (ps < FLAT_RATIO * rs))
        )
        flat2d |= is_flat
        del pg, rg, ps, rs, is_flat

    # Remove speckle: a pixel is flat if most of its neighbourhood is flat.
    flat2d = (uniform_filter(flat2d.astype(np.float32), 15, mode="constant") > 0.5) & rbm2d
    return flat2d.ravel()[gap_idx]


def smooth_masked(
    values: np.ndarray,
    mask: np.ndarray,
    sigma_px: float,
    factor: int = OFFSET_DOWNSAMPLE,
) -> np.ndarray:
    """
    Smooth regional field of `values` (2D) supported only where `mask` is True.
    Computed on a block-averaged grid, then interpolated back. Far from any
    support the field relaxes to the global mean, so there are no hard edges.
    """
    h, w = values.shape
    hc, wc = -(-h // factor), -(-w // factor)
    v = np.zeros((hc * factor, wc * factor), dtype=np.float32)
    m = np.zeros_like(v)
    v[:h, :w] = np.where(mask, values, 0.0)
    m[:h, :w] = mask
    v = v.reshape(hc, factor, wc, factor).sum(axis=(1, 3))
    m = m.reshape(hc, factor, wc, factor).sum(axis=(1, 3))
    if m.sum() <= 0:
        return np.zeros((h, w), dtype=np.float32)
    gmean = float(v.sum() / m.sum())
    s = max(sigma_px / factor, 1.0)
    vs = gaussian_filter(v, s, mode="constant")
    ms = gaussian_filter(m, s, mode="constant")
    lam = 0.1 * factor * factor
    res = (vs + lam * gmean) / (ms + lam)
    up = zoom(res, factor, order=1)
    return up[:h, :w].astype(np.float32)


# --------------------------------------------------------- spatial residual --
def spatial_fill_residual(
    cube: np.ndarray, source: np.ndarray, aoi_mask: np.ndarray
) -> int:
    """
    Fills AOI pixels still missing after every other stage from the nearest valid
    pixel, then smooths only those filled pixels. Works in place on cube (7,H,W)
    and source (H,W). Returns the number of pixels filled.
    """
    known = aoi_mask & np.all(np.isfinite(cube), axis=0)
    unknown = aoi_mask & ~known
    if not unknown.any() or not known.any():
        return 0

    iy, ix = distance_transform_edt(~known, return_distances=False, return_indices=True)
    uy, ux = iy[unknown], ix[unknown]
    weight = gaussian_filter(known.astype(np.float32), SPATIAL_SMOOTH_SIGMA)

    for b in range(cube.shape[0]):
        band = cube[b]
        nearest = band[uy, ux]
        smooth = gaussian_filter(np.where(known, band, 0.0).astype(np.float32), SPATIAL_SMOOTH_SIGMA)
        w = weight[unknown]
        smoothed = np.where(w > 0.02, smooth[unknown] / np.maximum(w, 1e-6), nearest)
        band[unknown] = smoothed.astype(np.float32)
        cube[b] = band

    source[unknown] = SRC_SPATIAL
    return int(unknown.sum())


# ------------------------------------------------------------------ output --
def derive_ndvi(reconstructed_cube: np.ndarray) -> np.ndarray:
    if reconstructed_cube.ndim != 3:
        raise ValueError("Reconstructed Landsat cube must have shape (bands, height, width).")
    if reconstructed_cube.shape[0] != EXPECTED_OUT_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_OUT_FEATURES} reconstructed bands, "
            f"got {reconstructed_cube.shape[0]}."
        )

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
    return np.clip(ndvi, -1.0, 1.0)


def write_raster(
    path: Path,
    array: np.ndarray,
    profile: Dict,
    nodata: float,
    dtype: str = "float32",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()

    if array.ndim == 2:
        height, width = array.shape
        count = 1
    elif array.ndim == 3:
        count, height, width = array.shape
    else:
        raise ValueError(f"Raster array must be 2D or 3D, got {array.shape}.")

    if height != output_profile["height"] or width != output_profile["width"]:
        raise ValueError("Raster dimensions do not match the target profile.")

    output_profile.update(
        {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": count,
            "dtype": dtype,
            "nodata": nodata,
            "compress": "deflate",
            "BIGTIFF": "IF_SAFER",
        }
    )

    with rasterio.open(path, "w", **output_profile) as dst:
        if dtype == "float32":
            output = np.where(np.isfinite(array), array, nodata).astype(np.float32, copy=False)
        else:
            output = array.astype(dtype, copy=False)
        if output.ndim == 2:
            dst.write(output, 1)
        else:
            dst.write(output)


def write_preview(
    path: Path,
    cube: np.ndarray,
    aoi_mask: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> None:
    """Quick-look RGB (Red, Green, Blue = bands 0-2), fixed stretch across months."""
    try:
        from PIL import Image
    except ImportError:
        return
    step = PREVIEW_STEP
    rgb = cube[0:3, ::step, ::step]
    top = (y_mean[:3] + 2.0 * y_std[:3]).reshape(3, 1, 1)
    img = np.clip(rgb / top, 0.0, 1.0) ** 0.6
    img = np.nan_to_num(img, nan=1.0)
    arr = (np.moveaxis(img, 0, -1) * 255).astype(np.uint8)
    arr[~aoi_mask[::step, ::step]] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


# ------------------------------------------------------------- one month --
def process_month(
    store: SceneStore,
    climatology: ClimatologyProvider,
    model: RBFN,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    processor: RasterProcessor,
    dem_flat: np.ndarray,
    aoi_mask: np.ndarray,
    target_profile: Dict,
    output_dir: Path,
    year: int,
    month: int,
    chunk_size: int,
    output_nodata: float,
    window: int,
    dt_per_month: float,
    max_dt_months: int,
    ood_dist: float,
    state: Dict,
    threshold_pct: float = 80.0,
) -> Dict:
    """`state` carries the previous month's filled image ("chain") and per-pixel
    age between calls, so months must be processed in chronological order."""
    tag = f"[{year}-{month:02d}]"
    height, width = aoi_mask.shape
    n_total = height * width
    aoi_flat = aoi_mask.ravel()
    n_aoi = int(aoi_flat.sum())

    # 1. REAL OBSERVATIONS of the target month
    target = store.get(year, month)
    tflat = target.reshape(N_BANDS, -1) if target is not None else None
    obs_flat = np.zeros(n_total, dtype=bool)
    if tflat is not None:
        obs_flat = aoi_flat & np.all(np.isfinite(tflat), axis=0)
    coverage_pct = 100.0 * obs_flat.sum() / max(n_aoi, 1)
    print(f"{tag} real coverage {coverage_pct:.2f}% ({'scene found' if target is not None else 'no scene'})")

    if coverage_pct >= threshold_pct and not PROCESS_SUFFICIENT_MONTHS:
        print(f"{tag} coverage >= {threshold_pct}%, skipped by configuration.")
        return {"status": "SKIPPED"}

    chain = state.get("chain")
    age_prev = state.get("age")

    out = np.full((N_BANDS, n_total), np.nan, dtype=np.float32)
    source = np.zeros(n_total, dtype=np.uint8)
    if tflat is not None and obs_flat.any():
        out[:, obs_flat] = tflat[:, obs_flat]
        source[obs_flat] = SRC_OBSERVED

    gap_idx = np.flatnonzero(aoi_flat & ~obs_flat)
    n_gap = gap_idx.size
    n_rbfn = n_guarded = n_clim = n_chain = n_chain_ctx = n_ood = 0
    age_hint = None

    if n_gap:
        # 2. RBFN prediction with per-pixel (real or chained) neighbours
        ctx = collect_temporal_context(
            store, year, month, gap_idx, window, chain, age_prev, dt_per_month, max_dt_months
        )
        n_chain_ctx = int(ctx["prev_filled"].sum())
        age_hint = np.where(
            ctx["prev_ok"] & ~ctx["prev_filled"], ctx["prev_dt"] / dt_per_month, -1.0
        ).astype(np.float32)

        pred_c = np.full((n_gap, N_BANDS), np.nan, dtype=np.float32)
        ood_c = np.zeros(n_gap, dtype=bool)
        sel = np.flatnonzero(ctx["prev_ok"] | ctx["next_ok"])
        if sel.size:
            try:
                era5_precip = sanitize_predictor(
                    processor.load_era5_predictor("precip", year, month),
                    x_mean[ERA5_PRECIP_INDEX], "ERA5 precip",
                )
                era5_temp = sanitize_predictor(
                    processor.load_era5_predictor("temp", year, month),
                    x_mean[ERA5_TEMP_INDEX], "ERA5 temp",
                )
                pred, ood = predict_gap_pixels(
                    model, x_mean, x_std, y_mean, y_std,
                    gap_idx, sel, ctx, era5_precip, era5_temp, dem_flat, chunk_size, ood_dist,
                )
                pred[~np.all(np.isfinite(pred), axis=1)] = np.nan
                pred_c[sel] = pred
                ood_c[sel] = ood
                del era5_precip, era5_temp, pred, ood
            except Exception as exc:
                print(f"{tag} ! RBFN skipped ({exc}); reference image used instead.")

        # 3. REFERENCE image with real texture: climatology > previous filled > neighbours
        ref_c = np.full((n_gap, N_BANDS), np.nan, dtype=np.float32)
        ref_src = np.zeros(n_gap, dtype=np.uint8)

        clim = climatology.get(month)
        if clim is not None:
            v = clim.reshape(N_BANDS, -1)[:, gap_idx].T
            ok = np.all(np.isfinite(v), axis=1)
            ref_c[ok] = v[ok]
            ref_src[ok] = SRC_CLIMATOLOGY
            del v, ok

        need = np.flatnonzero(ref_src == SRC_NONE)
        if chain is not None and need.size:
            v = chain.reshape(N_BANDS, -1)[:, gap_idx[need]].T
            ok = np.all(np.isfinite(v), axis=1)
            ref_c[need[ok]] = v[ok]
            ref_src[need[ok]] = SRC_CHAIN
            del v, ok

        need = np.flatnonzero(ref_src == SRC_NONE)
        if need.size:
            pok = ctx["prev_ok"][need][:, None]
            nok = ctx["next_ok"][need][:, None]
            wsum = pok.astype(np.float32) + nok.astype(np.float32)
            acc = ctx["prev_vals"][need] * pok + ctx["next_vals"][need] * nok
            ok = wsum[:, 0] > 0
            ref_c[need[ok]] = acc[ok] / wsum[ok]
            ref_src[need[ok]] = SRC_CHAIN
        del ctx

        has_ref = np.isfinite(ref_c[:, 0])
        pred_c[ood_c & ~has_ref] = np.nan  # out-of-distribution and nothing to anchor it
        rb = np.isfinite(pred_c[:, 0])

        # 4. GUARD: discard textureless predictions, bound the rest
        final_c = ref_c.copy()
        used = ref_src.copy()
        if rb.any():
            flat_c = np.zeros(n_gap, dtype=bool)
            if has_ref.any():
                flat_c = detect_flat(
                    pred_c, ref_c, gap_idx, rb & has_ref, (height, width), y_std
                )
                flat_c |= ood_c
            good = rb & has_ref & ~flat_c
            guarded = rb & has_ref & flat_c
            noref = rb & ~has_ref

            for b in range(N_BANDS):
                bound = MAX_ANOMALY_SIGMAS * float(y_std[b])
                r = ref_c[:, b]
                p = pred_c[:, b]

                # Regional offset: this month's real anomaly vs climatology, plus healthy RBFN pixels.
                off = None
                if guarded.any():
                    dgrid = np.full(n_total, np.nan, dtype=np.float32)
                    if clim is not None and tflat is not None and obs_flat.any():
                        cb = clim[b].ravel()
                        tb = tflat[b]
                        s = obs_flat & np.isfinite(cb) & np.isfinite(tb)
                        dgrid[s] = tb[s] - cb[s]
                    dgrid[gap_idx[good]] = p[good] - r[good]
                    dm = np.isfinite(dgrid)
                    if dm.any():
                        off = smooth_masked(
                            np.nan_to_num(dgrid).reshape(height, width),
                            dm.reshape(height, width),
                            OFFSET_SIGMA_PX,
                        ).ravel()[gap_idx]
                    del dgrid, dm
                if off is None:
                    off = np.zeros(n_gap, dtype=np.float32)

                final_c[good, b] = r[good] + np.clip(p[good] - r[good], -bound, bound)
                final_c[guarded, b] = r[guarded] + np.clip(off[guarded], -bound, bound)
                final_c[noref, b] = p[noref]

            used[good] = SRC_RBFN
            used[guarded] = SRC_GUARDED
            used[noref] = SRC_RBFN
            n_rbfn = int(good.sum() + noref.sum())
            n_guarded = int(guarded.sum())
            n_ood = int((ood_c & rb).sum())

        n_clim = int((used == SRC_CLIMATOLOGY).sum())
        n_chain = int((used == SRC_CHAIN).sum())

        out[:, gap_idx] = final_c.T
        source[gap_idx] = used
        del pred_c, ref_c, final_c, ref_src, used, has_ref, rb
        gc.collect()

    # 5. Range check, then SPATIAL fill of whatever is still empty
    cube = out.reshape(N_BANDS, height, width)
    lo, hi = store.lo, store.hi
    np.clip(cube, lo[:, None, None], hi[:, None, None], out=cube)
    cube[:, ~aoi_mask] = np.nan

    source2d = source.reshape(height, width)
    n_spatial = spatial_fill_residual(cube, source2d, aoi_mask)
    np.clip(cube, lo[:, None, None], hi[:, None, None], out=cube)
    cube[:, ~aoi_mask] = np.nan

    # 6. QA: share of the filled area that is still textureless
    gap_mask2d = (aoi_flat & ~obs_flat).reshape(height, width) & np.isfinite(cube[NIR_INDEX])
    qa_flat_pct = 0.0
    if gap_mask2d.any():
        sd = local_std(cube[NIR_INDEX], gap_mask2d, TEXTURE_WINDOW)
        okpix = gap_mask2d & np.isfinite(sd)
        if okpix.any():
            qa_flat_pct = float(
                100.0 * np.mean(sd[okpix] < FLAT_ABS_SIGMAS * float(y_std[NIR_INDEX]))
            )
        del sd, okpix
    del gap_mask2d

    # 7. Write
    ndvi = derive_ndvi(cube)
    band_path = output_dir / f"landsat_{year}_{month:02d}_rbfn_gapfilled.tif"
    ndvi_path = output_dir / f"ndvi_{year}_{month:02d}_rbfn_gapfilled.tif"
    prov_path = output_dir / "provenance" / f"landsat_{year}_{month:02d}_source.tif"
    write_raster(band_path, cube, target_profile, output_nodata)
    write_raster(ndvi_path, ndvi, target_profile, output_nodata)
    write_raster(prov_path, source2d, target_profile, 0, dtype="uint8")
    if WRITE_PREVIEWS:
        write_preview(output_dir / "preview" / f"landsat_{year}_{month:02d}.png", cube, aoi_mask, y_mean, y_std)

    final_valid = aoi_mask & np.all(np.isfinite(cube), axis=0)
    final_pct = 100.0 * final_valid.sum() / max(n_aoi, 1)
    n_obs = int((source == SRC_OBSERVED).sum())

    print(
        f"{tag} observed {n_obs:,} | RBFN {n_rbfn:,} | RBFN guarded {n_guarded:,} (out-of-distribution {n_ood:,}) | "
        f"climatology {n_clim:,} | previous-filled/neighbour {n_chain:,} | spatial {n_spatial:,} | "
        f"chain used as neighbour {n_chain_ctx:,}"
    )
    print(f"{tag} final coverage {final_pct:.2f}% | flat-block QA {qa_flat_pct:.2f}%")
    if qa_flat_pct > 5.0:
        print(f"{tag} ! WARNING: {qa_flat_pct:.1f}% of the filled area is still textureless. Inspect this month.")

    # 8. Update chain state for the next month
    if age_prev is None:
        age_new = np.full(n_total, window, dtype=np.int16)
    else:
        age_new = np.minimum(age_prev + 1, window).astype(np.int16)
    if age_hint is not None:
        has_real_prev = age_hint >= 0
        age_new[gap_idx[has_real_prev]] = np.minimum(age_hint[has_real_prev], window).astype(np.int16)
    age_new[(source == SRC_CLIMATOLOGY) | (source == SRC_SPATIAL)] = window
    age_new[obs_flat] = 0
    state["chain"] = cube
    state["age"] = age_new

    def pct(v: int) -> float:
        return round(100.0 * v / max(n_aoi, 1), 3)

    del out, ndvi, source, source2d
    gc.collect()

    return {
        "status": "COMPLETE" if (final_pct >= 99.9 and qa_flat_pct <= 5.0) else "CHECK",
        "year": year,
        "month": month,
        "total_aoi_pixels": n_aoi,
        "observed_pct": pct(n_obs),
        "rbfn_pct": pct(n_rbfn),
        "rbfn_guarded_pct": pct(n_guarded),
        "ood_pct": pct(n_ood),
        "climatology_pct": pct(n_clim),
        "prev_filled_or_neighbour_pct": pct(n_chain),
        "spatial_pct": pct(n_spatial),
        "chain_context_pct": pct(n_chain_ctx),
        "flat_block_qa_pct": round(qa_flat_pct, 3),
        "final_coverage_pct": round(float(final_pct), 3),
    }


# ------------------------------------------------------------------- main --
def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    config_path = root_dir / "src" / "config.yaml"
    config = load_config(config_path)

    seed = int(config["project"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    model, x_mean, x_std, y_mean, y_std = load_model_and_scalers(
        config=config, root_dir=root_dir
    )

    processor = RasterProcessor(config)
    dem = processor.load_dem()
    target_profile = processor.get_target_profile()
    dem_flat = sanitize_predictor(dem, x_mean[DEM_INDEX], "DEM")

    aoi_mask = get_aoi_mask(target_profile)

    gap_start, gap_end = config["landsat"]["gap_years"]
    chunk_size = int(config["model"]["rbfn"]["chunk_size"])
    output_nodata = float(config["gap_filling"]["output_nodata"])
    window = int(config["gap_filling"].get("search_window_months", SEARCH_WINDOW_MONTHS))
    dt_per_month, max_dt_months = infer_dt_settings(x_mean, x_std, window)
    ood_dist = rbf_ood_distance(model)
    print(
        f"RBFN: {model.num_centers} centres, gamma {float(model.gamma):.5f}; "
        f"out-of-distribution distance {ood_dist:.3f} (scaled units)."
    )
    output_dir = root_dir / config["paths"]["gapfilled_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    landsat_dir = root_dir / "data" / "landsat"
    index = index_scenes(landsat_dir)
    if not index:
        raise FileNotFoundError(f"No Landsat GeoTIFFs found in {landsat_dir}.")
    print(f"Indexed {len(index)} real Landsat scenes. Search window: +/-{window} months.")

    store = SceneStore(index, target_profile, y_mean, y_std)
    climatology = ClimatologyProvider(
        store=store,
        aoi_mask=aoi_mask,
        cache_dir=root_dir / config["paths"]["processed_dir"] / "climatology",
    )

    # Optional date range from the command line: START END as YYYY-MM.
    span = [a for a in sys.argv[1:] if re.fullmatch(r"\d{4}-\d{1,2}", a)]
    start_key, end_key = (int(gap_start), 1), (int(gap_end), 12)
    if len(span) >= 1:
        y, m = map(int, span[0].split("-"))
        start_key = (y, m)
        end_key = (y, m)
    if len(span) >= 2:
        y, m = map(int, span[1].split("-"))
        end_key = (y, m)
    print(f"Processing {start_key[0]}-{start_key[1]:02d} -> {end_key[0]}-{end_key[1]:02d}")

    state: Dict = {"chain": None, "age": None}
    rows = []
    counts = {"COMPLETE": 0, "CHECK": 0, "SKIPPED": 0, "FAILED": 0}

    for year in range(start_key[0], end_key[0] + 1):
        for month in range(1, 13):
            if not (start_key <= (year, month) <= end_key):
                continue
            print(f"Processing gap month: {year}-{month:02d}")
            try:
                result = process_month(
                    store=store,
                    climatology=climatology,
                    model=model,
                    x_mean=x_mean,
                    x_std=x_std,
                    y_mean=y_mean,
                    y_std=y_std,
                    processor=processor,
                    dem_flat=dem_flat,
                    aoi_mask=aoi_mask,
                    target_profile=target_profile,
                    output_dir=output_dir,
                    year=year,
                    month=month,
                    chunk_size=chunk_size,
                    output_nodata=output_nodata,
                    window=window,
                    dt_per_month=dt_per_month,
                    max_dt_months=max_dt_months,
                    ood_dist=ood_dist,
                    state=state,
                )
            except Exception as exc:
                traceback.print_exc()
                result = {"status": f"FAILED: {exc}", "year": year, "month": month}

            status = result["status"]
            counts[status if status in counts else "FAILED"] += 1
            if status != "SKIPPED":
                rows.append(result)
            gc.collect()

    summary_path = output_dir / "gapfill_run_summary.csv"
    fields = [
        "year", "month", "total_aoi_pixels", "observed_pct", "rbfn_pct",
        "rbfn_guarded_pct", "ood_pct", "climatology_pct", "prev_filled_or_neighbour_pct",
        "spatial_pct", "chain_context_pct", "flat_block_qa_pct",
        "final_coverage_pct", "status",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("Gap-filling inference completed.")
    print(f"Months complete: {counts['COMPLETE']}")
    print(f"Months to inspect (CHECK): {counts['CHECK']}")
    print(f"Months failed: {counts['FAILED']}")
    print(f"Months skipped: {counts['SKIPPED']}")
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    main()