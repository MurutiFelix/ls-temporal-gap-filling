# src/train.py

"""
Root training orchestrator for the Landsat Temporal Gap Filling pipeline.
Coordinates chronological dataset construction, training-only feature and
target scaling, RBF centre fitting, Ridge readout estimation, validation,
testing, and checkpoint persistence.

The RBFN training contract is:
Input features:
0:7 Previous Landsat bands
7:14 Next Landsat bands
14 Time distance to previous observation
15 Time distance to next observation
16 Previous observation availability
17 Next observation availability
18 ERA5 precipitation
19 ERA5 temperature
20 DEM
Total input features: 21
Total output features: 7 Landsat bands

The dataset is responsible for constructing the feature matrices and
chronological train/validation/test partitions. The trainer is responsible
for model-specific preprocessing and fitting.

Only the training partition is used to fit feature scalers, target scalers,
RBF centres, Gaussian gamma, and Ridge readout parameters.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

from src.models.train_rbfn import RBFNTrainer
from src.preprocessing.dataset import GapFillDataset

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CONFIG_PATH = SRC_DIR / "config.yaml"

EXPECTED_IN_FEATURES = 21
EXPECTED_OUT_FEATURES = 7

EXPECTED_LANDSAT_BANDS = (
    "Red",
    "Green",
    "Blue",
    "NIR",
    "SWIR1",
    "SWIR2",
    "Thermal",
)

EXPECTED_INPUT_FEATURES = (
    "landsat_prev",
    "landsat_next",
    "dt_prev",
    "dt_next",
    "prev_available",
    "next_available",
    "era5_precip",
    "era5_temp",
    "dem",
)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the central YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a valid dictionary."
        )

    return config


def validate_project_contract(
    config: Dict[str, Any],
) -> None:
    """
    Validate the configuration values required by the RBFN training pipeline.
    """
    landsat_config = config.get("landsat") or {}
    bands = landsat_config.get("bands") or []

    if tuple(bands) != EXPECTED_LANDSAT_BANDS:
        raise ValueError(
            "The configured Landsat band order does not match the locked "
            f"RBFN contract. Expected: {list(EXPECTED_LANDSAT_BANDS)}; "
            f"received: {bands}"
        )

    configured_num_bands = landsat_config.get("num_bands")

    if configured_num_bands != EXPECTED_OUT_FEATURES:
        raise ValueError(
            "landsat.num_bands must be 7 for the locked RBFN contract. "
            f"Found: {configured_num_bands}"
        )

    features = config.get("features") or {}
    inputs = features.get("inputs") or []

    if tuple(inputs) != EXPECTED_INPUT_FEATURES:
        raise ValueError(
            "The configured RBFN input feature order does not match the "
            f"locked contract. Expected: {list(EXPECTED_INPUT_FEATURES)}; "
            f"received: {inputs}"
        )


def validate_split(split: Any) -> None:
    """Validate the chronological dataset split dimensions and contents."""
    required_attributes = (
        "X_train",
        "X_val",
        "X_test",
        "Y_train",
        "Y_val",
        "Y_test",
        "train_months",
        "val_months",
        "test_months",
    )

    missing_attributes = [
        name
        for name in required_attributes
        if not hasattr(split, name)
    ]

    if missing_attributes:
        raise AttributeError(
            "Dataset split is missing required attributes: "
            + ", ".join(missing_attributes)
        )

    feature_shapes = {
        "training": split.X_train.shape,
        "validation": split.X_val.shape,
        "test": split.X_test.shape,
    }

    target_shapes = {
        "training": split.Y_train.shape,
        "validation": split.Y_val.shape,
        "test": split.Y_test.shape,
    }

    for partition, shape in feature_shapes.items():
        if len(shape) != 2:
            raise ValueError(
                f"{partition.capitalize()} feature matrix must be 2D; "
                f"received shape {shape}."
            )

        if shape[1] != EXPECTED_IN_FEATURES:
            raise ValueError(
                f"{partition.capitalize()} feature matrix contains "
                f"{shape[1]} features; expected "
                f"{EXPECTED_IN_FEATURES}."
            )

    for partition, shape in target_shapes.items():
        if len(shape) != 2:
            raise ValueError(
                f"{partition.capitalize()} target matrix must be 2D; "
                f"received shape {shape}."
            )

        if shape[1] != EXPECTED_OUT_FEATURES:
            raise ValueError(
                f"{partition.capitalize()} target matrix contains "
                f"{shape[1]} bands; expected "
                f"{EXPECTED_OUT_FEATURES}."
            )

    if len(split.train_months) == 0:
        raise ValueError(
            "Training partition contains no months."
        )

    if len(split.val_months) == 0:
        raise ValueError(
            "Validation partition contains no months."
        )

    if len(split.test_months) == 0:
        raise ValueError(
            "Test partition contains no months."
        )

    if split.X_train.shape[0] == 0:
        raise ValueError(
            "Training partition contains no valid pixels."
        )

    if split.X_val.shape[0] == 0:
        raise ValueError(
            "Validation partition contains no valid pixels."
        )

    if split.X_test.shape[0] == 0:
        raise ValueError(
            "Test partition contains no valid pixels."
        )


def print_split_summary(split: Any) -> None:
    """Print chronological dataset split information."""
    print("\nDataset Split Summary")
    print("-" * 60)
    print(
        f"Training months:   {len(split.train_months)} "
        f"| {split.train_months[0]} to {split.train_months[-1]}"
    )

    print(
        f"Validation months: {len(split.val_months)} "
        f"| {split.val_months[0]} to {split.val_months[-1]}"
    )

    print(
        f"Test months:       {len(split.test_months)} "
        f"| {split.test_months[0]} to {split.test_months[-1]}"
    )

    print()

    print(
        f"Training pixels:   {split.X_train.shape[0]:,}"
    )

    print(
        f"Validation pixels: {split.X_val.shape[0]:,}"
    )

    print(
        f"Test pixels:       {split.X_test.shape[0]:,}"
    )

    print()

    print(
        f"Input features:    {split.X_train.shape[1]}"
    )

    print(
        f"Output bands:      {split.Y_train.shape[1]}"
    )


def get_training_configuration(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract and validate RBFN training parameters from configuration."""
    model_config = config.get("model") or {}
    rbfn_config = model_config.get("rbfn") or {}

    batch_size = int(
        rbfn_config.get(
            "batch_size",
            4096,
        )
    )

    validation_ratio = float(
        rbfn_config.get(
            "validation_holdout_ratio",
            0.22,
        )
    )

    test_ratio = float(
        rbfn_config.get(
            "test_holdout_ratio",
            0.15,
        )
    )

    num_workers = int(
        rbfn_config.get(
            "num_workers",
            0,
        )
    )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_holdout_ratio must be between 0 and 1."
        )

    if not 0.0 < test_ratio < 1.0:
        raise ValueError(
            "test_holdout_ratio must be between 0 and 1."
        )

    if validation_ratio + test_ratio >= 1.0:
        raise ValueError(
            "Validation and test holdout ratios must sum to less than 1."
        )

    return {
        "batch_size": batch_size,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "num_workers": num_workers,
        "gamma": rbfn_config.get("gamma"),
    }


def build_dataset_split(
    config: Dict[str, Any],
    training_config: Dict[str, Any],
) -> Any:
    """Construct the chronological RBFN dataset split."""
    dataset = GapFillDataset(config)

    result = dataset.create_rbfn_dataloaders(
        batch_size=training_config["batch_size"],
        val_ratio=training_config["validation_ratio"],
        test_ratio=training_config["test_ratio"],
        num_workers=training_config["num_workers"],
    )

    if not isinstance(result, tuple) or len(result) != 4:
        raise ValueError(
            "GapFillDataset.create_rbfn_dataloaders() must return "
            "(train_loader, val_loader, test_loader, split)."
        )

    _, _, _, split = result

    validate_split(split)

    return split


def evaluate_partition(
    trainer: RBFNTrainer,
    X_scaled: Any,
    Y_scaled: Any,
    Y_raw: Any,
    partition_name: str,
) -> Dict[str, float]:
    """Evaluate one dataset partition and print its trainer metrics."""
    metrics = trainer.evaluate(
        X_eval_scaled=X_scaled,
        Y_eval_scaled=Y_scaled,
        Y_eval_raw=Y_raw,
    )

    print(f"\n{partition_name} Metrics")
    print("-" * 60)

    print(
        f"Scaled MSE:     "
        f"{metrics['eval_mse_scaled']:.8f}"
    )

    print(
        f"Physical RMSE:  "
        f"{metrics['eval_rmse_physical']:.8f}"
    )

    print(
        f"R²:             "
        f"{metrics['eval_r2']:.6f}"
    )

    return {
        key: float(value)
        for key, value in metrics.items()
    }


def save_training_outputs(
    trainer: RBFNTrainer,
    config: Dict[str, Any],
    split: Any,
    training_mse: float,
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    """Save the trained model, scalers, and reproducibility metadata."""
    paths_config = config.get("paths") or {}
    processed_dir = (
        ROOT_DIR
        / paths_config.get("processed_dir", "data/processed")
    )

    checkpoint_dir = processed_dir / "models"

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        checkpoint_dir
        / "rbfn_landsat_gap_filler.pt"
    )

    scaler_path = (
        checkpoint_dir
        / "rbfn_scalers.joblib"
    )

    gamma = (
        trainer.model.gamma
        .detach()
        .cpu()
        .item()
    )

    metadata = {
        "training_months": list(
            split.train_months
        ),
        "validation_months": list(
            split.val_months
        ),
        "test_months": list(
            split.test_months
        ),
        "in_features": EXPECTED_IN_FEATURES,
        "out_features": EXPECTED_OUT_FEATURES,
        "num_centers": int(
            trainer.num_centers
        ),
        "regularization_lambda": float(
            trainer.regularization_lambda
        ),
        "gamma": float(gamma),
        "chunk_size": int(
            trainer.chunk_size
        ),
        "training_scaled_mse": float(
            training_mse
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    trainer.save_checkpoint(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        metadata=metadata,
    )

    print("\nSaved Training Outputs")
    print("-" * 60)

    print(
        f"Model checkpoint:  {model_path}"
    )

    print(
        f"Scaler checkpoint: {scaler_path}"
    )


def main() -> None:
    """Execute the complete RBFN training pipeline."""
    config = load_config(
        CONFIG_PATH
    )

    validate_project_contract(
        config
    )

    print("=" * 70)
    print("LANDSAT TEMPORAL GAP FILLING")
    print("RBFN TRAINING PIPELINE")
    print("=" * 70)

    training_config = (
        get_training_configuration(
            config
        )
    )

    print(
        "\n1. Building chronological dataset"
    )

    split = build_dataset_split(
        config=config,
        training_config=training_config,
    )

    print_split_summary(
        split
    )

    print(
        "\n2. Initializing RBFN trainer"
    )

    trainer = RBFNTrainer(
        config=config,
        in_features=EXPECTED_IN_FEATURES,
        out_features=EXPECTED_OUT_FEATURES,
    )

    print(
        f"Device:               {trainer.device}"
    )

    print(
        f"RBF centres:          {trainer.num_centers}"
    )

    print(
        f"Regularization lambda: "
        f"{trainer.regularization_lambda}"
    )

    print(
        f"Chunk size:           "
        f"{trainer.chunk_size:,}"
    )

    print(
        "\n3. Fitting training-only scalers"
    )

    X_train_scaled, Y_train_scaled = (
        trainer.fit_scalers(
            split.X_train,
            split.Y_train,
        )
    )

    X_val_scaled = (
        trainer.transform_features(
            split.X_val
        )
    )

    Y_val_scaled = (
        trainer.transform_targets(
            split.Y_val
        )
    )

    X_test_scaled = (
        trainer.transform_features(
            split.X_test
        )
    )

    Y_test_scaled = (
        trainer.transform_targets(
            split.Y_test
        )
    )

    print(
        "Training feature and target scalers fitted."
    )

    print(
        "\n4. Fitting RBF centres and Ridge readout"
    )

    training_mse = trainer.fit_ridge(
        X_train_scaled=X_train_scaled,
        Y_train_scaled=Y_train_scaled,
        user_gamma=training_config["gamma"],
    )

    print(
        f"Training scaled MSE: "
        f"{training_mse:.8f}"
    )

    print(
        "\n5. Evaluating validation partition"
    )

    validation_metrics = evaluate_partition(
        trainer=trainer,
        X_scaled=X_val_scaled,
        Y_scaled=Y_val_scaled,
        Y_raw=split.Y_val,
        partition_name="Validation",
    )

    print(
        "\n6. Evaluating final test partition"
    )

    test_metrics = evaluate_partition(
        trainer=trainer,
        X_scaled=X_test_scaled,
        Y_scaled=Y_test_scaled,
        Y_raw=split.Y_test,
        partition_name="Test",
    )

    print(
        "\n7. Saving model checkpoint"
    )

    save_training_outputs(
        trainer=trainer,
        config=config,
        split=split,
        training_mse=training_mse,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RBFN TRAINING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()