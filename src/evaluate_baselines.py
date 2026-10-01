# src/evaluate_baselines.py
"""Compare the trained RBFN with simple temporal baselines on validation/test months."""

import joblib
import numpy as np
import torch

from src.models.train_rbfn import RBFNTrainer
from src.preprocessing.dataset import GapFillDataset
from src.train import CONFIG_PATH, ROOT_DIR, load_config

PROCESSED = ROOT_DIR / "data" / "processed"
CACHE_PATH = PROCESSED / "cache" / "eval_split.npz"  # delete to force a rebuild
MODEL_PATH = PROCESSED / "models" / "rbfn_landsat_gap_filler.pt"
SCALER_PATH = PROCESSED / "models" / "rbfn_scalers.joblib"

PREV, NEXT = slice(0, 7), slice(7, 14)
DT_PREV, DT_NEXT, HAS_PREV, HAS_NEXT = 14, 15, 16, 17


def get_split(config):
    # build the chronological split once, then reuse it from disk
    if CACHE_PATH.exists():
        return dict(np.load(CACHE_PATH))
    ds = GapFillDataset(config)
    split = ds.create_temporal_holdout(ds.build_monthly_samples())
    keys = ("X_train", "Y_train", "X_val", "Y_val", "X_test", "Y_test")
    arrays = {k: getattr(split, k) for k in keys}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, **arrays)
    return arrays


def load_trainer():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    trainer = RBFNTrainer(ckpt["config"], ckpt["in_features"], ckpt["out_features"])
    state = ckpt["state_dict"]
    trainer.model.centers = state["centers"].to(trainer.device)  # buffer starts empty
    trainer.model.load_state_dict(state)
    scalers = joblib.load(SCALER_PATH)
    trainer.x_mean, trainer.x_std = scalers["x_mean"], scalers["x_std"]
    trainer.y_mean, trainer.y_std = scalers["y_mean"], scalers["y_std"]
    trainer._fitted = True
    return trainer


def fit_linear(X, Y, lam=1.0, chunk=1_000_000):
    # ridge on the same 21 features, solved from chunked normal equations
    mean = X.mean(0, dtype=np.float64)
    std = X.std(0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    y_mean = Y.mean(0, dtype=np.float64)
    d = X.shape[1]
    xtx, xty = np.zeros((d, d)), np.zeros((d, Y.shape[1]))
    for i in range(0, len(X), chunk):
        xs = (X[i:i + chunk] - mean) / std
        xtx += xs.T @ xs
        xty += xs.T @ (Y[i:i + chunk] - y_mean)
    w = np.linalg.solve(xtx + lam * np.eye(d), xty)
    return lambda Xe: ((Xe - mean) / std) @ w + y_mean


def temporal_baselines(X):
    prev, nxt = X[:, PREV], X[:, NEXT]
    dtp, dtn = X[:, DT_PREV], X[:, DT_NEXT]
    hp, hn = X[:, HAS_PREV] == 1, X[:, HAS_NEXT] == 1
    # persistence: copy the nearest available scene (NaN if there is none)
    use_prev = hp & (~hn | (dtp <= dtn))
    nearest = np.where(use_prev[:, None], prev, nxt)
    nearest[~(hp | hn)] = np.nan
    # interpolation: the closer observation gets the larger weight
    w = (dtn / (dtp + dtn))[:, None]
    interp = np.where((hp & hn)[:, None], w * prev + (1 - w) * nxt, nearest)
    return nearest, interp


def score(Y, P, y_std):
    mse = np.mean((Y - P) ** 2, axis=0)
    # nmse uses the training target std, so it matches the trainer's scaled MSE
    return {
        "nmse": float(np.mean(mse / y_std ** 2)),
        "r2": 1 - mse / np.var(Y, axis=0),
    }


def report(name, X, Y, preds, y_std):
    hp, hn = X[:, HAS_PREV] == 1, X[:, HAS_NEXT] == 1
    print(f"\n=== {name} === rows with no neighbour: {int((~(hp | hn)).sum())}")
    print(f"rbfn NMSE on all rows: {score(Y, preds['rbfn'], y_std)['nmse']:.4f}")
    subsets = {"any neighbour": hp | hn, "two-sided": hp & hn, "one-sided": hp ^ hn}
    for sname, m in subsets.items():
        print(f"\n{sname}: {int(m.sum()):,} rows (NMSE | per-band R2)")
        for method, P in preds.items():
            s = score(Y[m], P[m], y_std)
            r2 = " ".join(f"{v:6.3f}" for v in s["r2"])
            print(f"  {method:<13} {s['nmse']:.4f} | {r2}")

    # error against time distance to the nearest observation
    nearest_dt = np.minimum(
        np.where(hp, X[:, DT_PREV], np.inf), np.where(hn, X[:, DT_NEXT], np.inf)
    )
    bins = np.digitize(nearest_dt, [2, 3, 7])  # 1 | 2 | 3-6 | 7+ months
    print("\nNMSE by distance to nearest observation")
    for b, label in enumerate(["1 mo", "2 mo", "3-6 mo", "7+ mo"]):
        m = (hp | hn) & (bins == b)
        if m.any():
            cells = " ".join(f"{k}={score(Y[m], P[m], y_std)['nmse']:.4f}" for k, P in preds.items())
            print(f"  {label:<7} n={int(m.sum()):>9,} {cells}")


def main():
    config = load_config(CONFIG_PATH)
    data = get_split(config)
    trainer = load_trainer()
    y_std = np.asarray(trainer.y_std, dtype=np.float64).ravel()
    linear = fit_linear(data["X_train"], data["Y_train"])
    print("Band order:", ", ".join(config["landsat"]["bands"]))

    for name in ("val", "test"):
        X, Y = data[f"X_{name}"], data[f"Y_{name}"]
        nearest, interp = temporal_baselines(X)
        preds = {
            "persistence": nearest,
            "interpolation": interp,
            "linear": linear(X),
            "rbfn": trainer.predict(X),
        }
        report(name, X, Y, preds, y_std)


if __name__ == "__main__":
    main()