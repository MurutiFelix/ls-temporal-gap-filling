# src/models/tuning.py

"""
Hyperparameter tuning module for the Radial Basis Function Network (RBFN).
Performs exhaustive grid search over:

- Number of RBF centres
- Ridge regularization strength
- Gamma multiplier
using a strict chronological:

Training -> Validation -> Test
partition.
Scalers, K-Means centres, gamma, and Ridge weights are fitted using the
training split only. The validation split is used exclusively for
hyperparameter selection. The test split remains untouched until the
selected configuration has been evaluated for final reporting.
"""
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
import itertools
import json
import math
from src.models.train_rbfn import RBFNTrainer
from src.preprocessing.dataset import GapFillDataset

EXPECTED_IN_FEATURES = 21
EXPECTED_OUT_FEATURES = 7


class RBFNHyperparameterTuner:
    """
    Performs systematic grid-search hyperparameter optimization for RBFN.
    """
    def __init__(
        self,
        base_config: dict,
    ):
        self.base_config = base_config

        output_root = Path(
            base_config["paths"]["outputs_dir"]
        )

        self.output_dir = output_root / "tuning"

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _generate_param_grid(
        self,
        search_space: Dict[str, List[Any]],
    ) -> List[Dict[str, Any]]:
        """
        Expand parameter lists into exhaustive combinations.
        """
        if not search_space:
            raise ValueError(
                "search_space cannot be empty."
            )

        keys = list(search_space.keys())
        values = list(search_space.values())

        if any(
            not isinstance(parameter_values, list)
            or not parameter_values
            for parameter_values in values
        ):
            raise ValueError(
                "Every search-space parameter must contain "
                "a non-empty list of candidate values."
            )

        return [
            dict(zip(keys, combination))
            for combination in itertools.product(
                *values
            )
        ]

    def _get_default_search_space(
        self,
    ) -> Dict[str, List[Any]]:
        """
        Return the default RBFN hyperparameter search space.
        """
        return {
            "num_centers": [
                25,
                50,
                100,
                200,
            ],
            "regularization_lambda": [
                1e-4,
                1e-3,
                1e-2,
                1e-1,
            ],
            "gamma_multiplier": [
                0.2,
                0.5,
                1.0,
                2.0,
            ],
        }

    def _get_split_ratios(
        self,
        val_ratio: Optional[float],
        test_ratio: Optional[float],
    ) -> tuple[float, float]:
        """
        Resolve temporal holdout ratios from explicit arguments or config.
        """
        rbfn_config = self.base_config["model"]["rbfn"]

        if val_ratio is None:
            val_ratio = float(
                rbfn_config["validation_holdout_ratio"]
            )

        if test_ratio is None:
            test_ratio = float(
                rbfn_config["test_holdout_ratio"]
            )

        if not 0.0 < val_ratio < 1.0:
            raise ValueError(
                "val_ratio must be greater than 0 and less than 1."
            )

        if not 0.0 < test_ratio < 1.0:
            raise ValueError(
                "test_ratio must be greater than 0 and less than 1."
            )

        if val_ratio + test_ratio >= 1.0:
            raise ValueError(
                "val_ratio + test_ratio must be less than 1."
            )

        return val_ratio, test_ratio

    def run_grid_search(
        self,
        search_space: Optional[
            Dict[str, List[Any]]
        ] = None,
        test_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute exhaustive RBFN grid search.

        The default search space contains:

            num_centers:
                [25, 50, 100, 200]

            regularization_lambda:
                [1e-4, 1e-3, 1e-2, 1e-1]

            gamma_multiplier:
                [0.2, 0.5, 1.0, 2.0]

        This produces 64 configurations.

        Model selection criterion:

            Minimum validation RMSE on the physical target scale.

        The test split is evaluated only after the best configuration has
        been selected using validation data.
        """
        if search_space is None:
            search_space = self._get_default_search_space()

        val_ratio, test_ratio = self._get_split_ratios(
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        param_combinations = self._generate_param_grid(
            search_space
        )

        print(
            "Starting RBFN Grid Search "
            f"across {len(param_combinations)} configurations..."
        )

        dataset_builder = GapFillDataset(
            self.base_config
        )

        monthly_samples = (
            dataset_builder.build_monthly_samples()
        )

        if not monthly_samples:
            raise RuntimeError(
                "No valid monthly training samples were generated."
            )

        split = (
            dataset_builder.create_temporal_holdout(
                monthly_samples=monthly_samples,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
        )

        if len(split.X_train) == 0:
            raise RuntimeError(
                "The training split contains no samples."
            )

        if len(split.X_val) == 0:
            raise RuntimeError(
                "The validation split contains no samples."
            )

        if len(split.X_test) == 0:
            raise RuntimeError(
                "The test split contains no samples."
            )

        if split.X_train.shape[1] != EXPECTED_IN_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_IN_FEATURES} input features, "
                f"received {split.X_train.shape[1]}."
            )

        if split.Y_train.shape[1] != EXPECTED_OUT_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_OUT_FEATURES} output bands, "
                f"received {split.Y_train.shape[1]}."
            )

        print("\nTemporal split:")
        print(
            f"  Training months: "
            f"{split.train_months[0]} -> "
            f"{split.train_months[-1]}"
        )
        print(
            f"  Validation months: "
            f"{split.val_months[0]} -> "
            f"{split.val_months[-1]}"
        )
        print(
            f"  Test months: "
            f"{split.test_months[0]} -> "
            f"{split.test_months[-1]}"
        )
        print(
            f"  Training samples: {len(split.X_train)}"
        )
        print(
            f"  Validation samples: {len(split.X_val)}"
        )
        print(
            f"  Test samples: {len(split.X_test)}"
        )

        in_features = split.X_train.shape[1]
        out_features = split.Y_train.shape[1]

        results: List[Dict[str, Any]] = []

        best_val_score = float("inf")
        best_params: Optional[Dict[str, Any]] = None
        best_trainer: Optional[RBFNTrainer] = None

        for idx, params in enumerate(
            param_combinations,
            start=1,
        ):
            trial_config = deepcopy(
                self.base_config
            )

            trial_config.setdefault(
                "model",
                {},
            )

            trial_config["model"].setdefault(
                "rbfn",
                {},
            )

            trial_config["model"]["rbfn"].update(
                params
            )

            trainer = RBFNTrainer(
                config=trial_config,
                in_features=in_features,
                out_features=out_features,
            )

            (
                X_train_scaled,
                Y_train_scaled,
            ) = trainer.fit_scalers(
                split.X_train,
                split.Y_train,
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

            train_mse_scaled = (
                trainer.fit_ridge(
                    X_train_scaled,
                    Y_train_scaled,
                )
            )

            val_metrics = trainer.evaluate(
                X_eval_scaled=X_val_scaled,
                Y_eval_scaled=Y_val_scaled,
                Y_eval_raw=split.Y_val,
            )

            val_rmse = float(
                val_metrics["eval_rmse_physical"]
            )

            fitted_gamma = float(
                trainer.model.gamma.item()
            )

            record = {
                "trial_id": idx,
                "params": params,
                "fitted_gamma": fitted_gamma,
                "train_mse_scaled": float(
                    train_mse_scaled
                ),
                "val_metrics": val_metrics,
            }

            results.append(record)

            print(
                f"[{idx}/{len(param_combinations)}] "
                f"K={params['num_centers']} | "
                f"Lambda={params['regularization_lambda']:.1e} | "
                f"Gamma multiplier="
                f"{params['gamma_multiplier']:.2f} | "
                f"Gamma={fitted_gamma:.6f} | "
                f"Val RMSE={val_rmse:.6f}"
            )

            if (
                math.isfinite(val_rmse)
                and val_rmse < best_val_score
            ):
                best_val_score = val_rmse
                best_params = deepcopy(
                    params
                )
                best_trainer = trainer

        if best_trainer is None or best_params is None:
            raise RuntimeError(
                "Grid search failed to identify a valid model."
            )

        print(
            "\nRunning final evaluation on "
            "the untouched test set..."
        )

        X_test_scaled = (
            best_trainer.transform_features(
                split.X_test
            )
        )

        Y_test_scaled = (
            best_trainer.transform_targets(
                split.Y_test
            )
        )

        test_metrics = best_trainer.evaluate(
            X_eval_scaled=X_test_scaled,
            Y_eval_scaled=Y_test_scaled,
            Y_eval_raw=split.Y_test,
        )

        final_test_rmse = float(
            test_metrics["eval_rmse_physical"]
        )

        print(
            f"Final Test RMSE: "
            f"{final_test_rmse:.6f}"
        )

        summary = {
            "num_trials": len(
                param_combinations
            ),
            "search_space": search_space,
            "split": {
                "train_ratio": (
                    len(split.train_months)
                    / len(monthly_samples)
                ),
                "validation_ratio": (
                    len(split.val_months)
                    / len(monthly_samples)
                ),
                "test_ratio": (
                    len(split.test_months)
                    / len(monthly_samples)
                ),
                "configured_validation_ratio": val_ratio,
                "configured_test_ratio": test_ratio,
                "train_months": split.train_months,
                "val_months": split.val_months,
                "test_months": split.test_months,
            },
            "best_params": best_params,
            "best_fitted_gamma": float(
                best_trainer.model.gamma.item()
            ),
            "best_val_rmse_physical": (
                best_val_score
            ),
            "final_test_metrics": test_metrics,
            "all_trials": results,
        }

        summary_path = (
            self.output_dir
            / "tuning_summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
                allow_nan=False,
            )

        best_trainer.save_checkpoint(
            model_path=str(
                self.output_dir
                / "best_rbfn_model.pt"
            ),
            scaler_path=str(
                self.output_dir
                / "best_rbfn_scalers.joblib"
            ),
            metadata={
                "tuning_summary": summary,
                "selected_for_final_evaluation": True,
                "train_months": split.train_months,
                "val_months": split.val_months,
                "test_months": split.test_months,
            },
        )

        print(
            "\nRBFN tuning complete."
        )
        print(
            f"Best validation RMSE: "
            f"{best_val_score:.6f}"
        )
        print(
            f"Final test RMSE: "
            f"{final_test_rmse:.6f}"
        )
        print(
            f"Best parameters: "
            f"{best_params}"
        )
        print(
            f"Tuning summary saved to: "
            f"{summary_path}"
        )

        return summary