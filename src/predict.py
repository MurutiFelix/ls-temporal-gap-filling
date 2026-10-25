# src/predict.py

"""
RBFN inference engine for historical Landsat temporal gap filling (v2).

Goal: every month in the gap period leaves as a complete, seamless 7-band
Landsat cube (plus NDVI) with no empty months, no partial months and no
constant-value blocks.

Fill hierarchy (each AOI pixel is filled by the first stage that can serve it):

  1. OBSERVED     real Landsat pixels of the target month are kept untouched.
  2. RBFN         gap pixels that have a real previous and/or next Landsat
                  observation within the search window. Temporal context is
                  searched PER PIXEL (nearest valid previous / next scene), so
                  SLC-off stripes, clouds and partial tiles are all handled by
                  the same rule, and dt_prev / dt_next are per-pixel values
                  that are zero exactly when the matching availability flag is 0
                  (the same convention the model was trained with).
  3. CLIMATOLOGY  pixels with no real observation inside the window (typical
                  for 1996-1999, where most of the catchment has no Landsat
                  archive). Uses the per-pixel, 30 m multi-year composite of the
                  same calendar month built from real scenes, so it keeps real
                  spatial texture instead of a constant value.
  4. SPATIAL      any residual pixel that was never observed in any year is
                  filled from the nearest valid pixel and lightly smoothed.

Why the old script produced empty / partial / blocky months
-----------------------------------------------------------
  * A scene-level 80 % coverage gate disabled RBFN for every neighbour that was
    only partly valid. In 1995-1999 no neighbour reaches 80 %, so RBFN never ran.
  * Fallback pixels were then taken from y_mean (one constant per band, i.e. a
    uniform block) because the climatology file search pattern ("_MM_") did not
    match stems that end in the month, so the composite was always None.
  * Months with >= 80 % coverage were skipped completely, leaving residual gaps.
  * Pixels with neither neighbour were still sent to RBFN with ERA5 + DEM only,
    and ERA5 is a coarse grid, which gives rectangular blocks.
  * dt_prev / dt_next were non-zero for pixels whose availability flag was 0.
  * Any NaN in ERA5 / DEM removed the pixel from valid_mask, which is why some
    months were written completely empty.
  * Reflectance scaling was applied in the climatology path only, and outputs
    were never range-checked (hence the repeated 1.6022 maximum).

Extra outputs per month
-----------------------
  landsat_YYYY_MM_rbfn_gapfilled.tif        7 bands, config order
  ndvi_YYYY_MM_rbfn_gapfilled.tif           NDVI
  provenance/landsat_YYYY_MM_source.tif     uint8: 1 observed, 2 RBFN,
                                            3 climatology, 4 spatial
  gapfill_run_summary.csv                   coverage and source shares per month
"""
import csv
import gc
import re
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
from scipy.ndimage import distance_transform_edt, gaussian_filter

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
# Months searched either side of the target for a real observation. Can be
# overridden with config["gap_filling"]["search_window_months"].
SEARCH_WINDOW_MONTHS = 12
# dt is expressed in months (the old script defaulted dt to 12.0). If the model
# was trained with dt in days, set this to 30.4.
DT_PER_MONTH = 1.0
# True: also fill the residual gaps of months that already have >= 80 % real
# coverage, so every month of the series is complete.
PROCESS_SUFFICIENT_MONTHS = True
# Number of full scenes kept in memory (each is 7 x H x W float32).
SCENE_CACHE_SIZE = 4
# Output values are clipped to y_mean +/- CLIP_SIGMAS * y_std (lower bound >= 0).
CLIP_SIGMAS = 8.0
# Smoothing (pixels) applied only to spatially filled residual pixels.
SPATIAL_SMOOTH_SIGMA = 3.0

# Provenance codes
SRC_NONE = 0
SRC_OBSERVED = 1
SRC_RBFN = 2
SRC_CLIMATOLOGY = 3
SRC_SPATIAL = 4

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

    model = RBFN(
        in_features=EXPECTED_IN_FEATURES,
        num_centers=int(rbfn_config["num_centers"]),
        out_features=EXPECTED_OUT_FEATURES,
        rbf_chunk_size=int(rbfn_config["chunk_size"]),
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

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


# ------------------------------------------------- per-pixel temporal context --
def collect_temporal_context(
    store: SceneStore,
    year: int,
    month: int,
    gap_idx: np.ndarray,
    window: int,
) -> Dict[str, np.ndarray]:
    """
    For every gap pixel, finds the nearest real previous and the nearest real
    next observation within `window` months. Values, dt and availability are all
    per pixel. Where nothing is found, bands stay zero, dt stays 0 and the flag
    stays 0, exactly the convention used in training.
    """
    n = gap_idx.size
    ctx = {
        "prev_vals": np.zeros((n, N_BANDS), dtype=np.float32),
        "next_vals": np.zeros((n, N_BANDS), dtype=np.float32),
        "prev_dt": np.zeros(n, dtype=np.float32),
        "next_dt": np.zeros(n, dtype=np.float32),
        "prev_ok": np.zeros(n, dtype=bool),
        "next_ok": np.zeros(n, dtype=bool),
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
                ctx[f"{direction}_dt"][sel] = k * DT_PER_MONTH
                ok[sel] = True
        if ctx["prev_ok"].all() and ctx["next_ok"].all():
            break
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
) -> np.ndarray:
    """RBFN inference for gap_idx[sel], built and scaled chunk by chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    n = sel.size
    out = np.empty((n, EXPECTED_OUT_FEATURES), dtype=np.float32)

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
            pred_scaled = model(torch.from_numpy(X_scaled)).cpu().numpy()
            out[start:end] = pred_scaled * y_std + y_mean

    return out


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
    threshold_pct: float = 80.0,
) -> Dict:
    tag = f"[{year}-{month:02d}]"
    height, width = aoi_mask.shape
    aoi_flat = aoi_mask.ravel()
    n_aoi = int(aoi_flat.sum())

    # 1. REAL OBSERVATIONS of the target month
    target = store.get(year, month)
    obs_flat = np.zeros(aoi_flat.shape, dtype=bool)
    if target is not None:
        obs_flat = aoi_flat & np.all(np.isfinite(target.reshape(N_BANDS, -1)), axis=0)
    coverage_pct = 100.0 * obs_flat.sum() / max(n_aoi, 1)
    print(f"{tag} real coverage {coverage_pct:.2f}% ({'scene found' if target is not None else 'no scene'})")

    if coverage_pct >= threshold_pct and not PROCESS_SUFFICIENT_MONTHS:
        print(f"{tag} coverage >= {threshold_pct}%, skipped by configuration.")
        return {"status": "SKIPPED"}

    out = np.full((N_BANDS, height * width), np.nan, dtype=np.float32)
    source = np.zeros(height * width, dtype=np.uint8)

    if target is not None and obs_flat.any():
        out[:, obs_flat] = target.reshape(N_BANDS, -1)[:, obs_flat]
        source[obs_flat] = SRC_OBSERVED

    gap_idx = np.flatnonzero(aoi_flat & ~obs_flat)

    # 2. RBFN where a real previous and/or next observation exists per pixel
    n_rbfn = 0
    if gap_idx.size:
        ctx = collect_temporal_context(store, year, month, gap_idx, window)
        has_ctx = ctx["prev_ok"] | ctx["next_ok"]
        sel = np.flatnonzero(has_ctx)
        if sel.size:
            era5_precip = sanitize_predictor(
                processor.load_era5_predictor("precip", year, month),
                x_mean[ERA5_PRECIP_INDEX], "ERA5 precip",
            )
            era5_temp = sanitize_predictor(
                processor.load_era5_predictor("temp", year, month),
                x_mean[ERA5_TEMP_INDEX], "ERA5 temp",
            )
            pred = predict_gap_pixels(
                model, x_mean, x_std, y_mean, y_std,
                gap_idx, sel, ctx, era5_precip, era5_temp, dem_flat, chunk_size,
            )
            finite = np.all(np.isfinite(pred), axis=1)
            px = gap_idx[sel[finite]]
            out[:, px] = pred[finite].T
            source[px] = SRC_RBFN
            n_rbfn = int(finite.sum())
            del era5_precip, era5_temp, pred, finite, px
        del ctx, has_ctx, sel

    # 3. CLIMATOLOGY for pixels with no real observation inside the window
    remaining = gap_idx[source[gap_idx] == SRC_NONE] if gap_idx.size else gap_idx
    n_clim = 0
    if remaining.size:
        clim = climatology.get(month)
        if clim is not None:
            vals = clim.reshape(N_BANDS, -1)[:, remaining].T
            ok = np.all(np.isfinite(vals), axis=1)
            out[:, remaining[ok]] = vals[ok].T
            source[remaining[ok]] = SRC_CLIMATOLOGY
            n_clim = int(ok.sum())
            del vals, ok

    # 4. Range check, then SPATIAL fill of whatever is still empty
    cube = out.reshape(N_BANDS, height, width)
    lo, hi = store.lo, store.hi
    np.clip(cube, lo[:, None, None], hi[:, None, None], out=cube)
    cube[:, ~aoi_mask] = np.nan

    source2d = source.reshape(height, width)
    n_spatial = spatial_fill_residual(cube, source2d, aoi_mask)
    np.clip(cube, lo[:, None, None], hi[:, None, None], out=cube)
    cube[:, ~aoi_mask] = np.nan

    # 5. Write
    ndvi = derive_ndvi(cube)

    band_path = output_dir / f"landsat_{year}_{month:02d}_rbfn_gapfilled.tif"
    ndvi_path = output_dir / f"ndvi_{year}_{month:02d}_rbfn_gapfilled.tif"
    prov_path = output_dir / "provenance" / f"landsat_{year}_{month:02d}_source.tif"

    write_raster(band_path, cube, target_profile, output_nodata)
    write_raster(ndvi_path, ndvi, target_profile, output_nodata)
    write_raster(prov_path, source2d, target_profile, 0, dtype="uint8")

    final_valid = aoi_mask & np.all(np.isfinite(cube), axis=0)
    final_pct = 100.0 * final_valid.sum() / max(n_aoi, 1)
    n_obs = int((source == SRC_OBSERVED).sum())

    print(
        f"{tag} observed {n_obs:,} | RBFN {n_rbfn:,} | climatology {n_clim:,} | "
        f"spatial {n_spatial:,} | final coverage {final_pct:.2f}%"
    )
    red_mean = float(np.nanmean(cube[RED_INDEX][final_valid])) if final_valid.any() else float("nan")
    blue_mean = float(np.nanmean(cube[2][final_valid])) if final_valid.any() else float("nan")
    print(f"{tag} sanity -> Red {red_mean:.4f} | Blue {blue_mean:.4f}")

    def pct(v: int) -> float:
        return round(100.0 * v / max(n_aoi, 1), 3)

    del out, cube, ndvi, source, source2d, target
    gc.collect()

    return {
        "status": "COMPLETE" if final_pct >= 99.9 else "PARTIAL",
        "year": year,
        "month": month,
        "total_aoi_pixels": n_aoi,
        "observed_pct": pct(n_obs),
        "rbfn_pct": pct(n_rbfn),
        "climatology_pct": pct(n_clim),
        "spatial_pct": pct(n_spatial),
        "final_coverage_pct": round(final_pct, 3),
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

    rows = []
    total_complete = 0
    total_partial = 0
    total_failed = 0
    total_skipped = 0

    for year in range(int(gap_start), int(gap_end) + 1):
        for month in range(1, 13):
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
                )
            except Exception as exc:
                traceback.print_exc()
                result = {"status": f"FAILED: {exc}", "year": year, "month": month}

            status = result["status"]
            if status == "COMPLETE":
                total_complete += 1
            elif status == "PARTIAL":
                total_partial += 1
            elif status == "SKIPPED":
                total_skipped += 1
            else:
                total_failed += 1

            if status != "SKIPPED":
                rows.append(result)
            gc.collect()

    summary_path = output_dir / "gapfill_run_summary.csv"
    fields = [
        "year", "month", "total_aoi_pixels", "observed_pct", "rbfn_pct",
        "climatology_pct", "spatial_pct", "final_coverage_pct", "status",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("Gap-filling inference completed.")
    print(f"Months complete (>= 99.9% of AOI): {total_complete}")
    print(f"Months partial: {total_partial}")
    print(f"Months failed: {total_failed}")
    print(f"Months skipped: {total_skipped}")
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    main()