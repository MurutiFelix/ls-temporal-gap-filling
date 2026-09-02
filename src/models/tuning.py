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

The training set is used to fit scalers, K-Means centres, gamma and Ridge
weights.

The validation set is used exclusively for hyperparameter/model selection.

The test set remains completely untouched until the best configuration has
been selected and is evaluated exactly once for final reporting.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
import itertools
import json
import math

from src.models.train_rbfn import RBFNTrainer
from src.preprocessing.dataset import GapFillDataset


class RBFNHyperparameterTuner:
    """
    Performs systematic grid-search hyperparameter optimization for RBFN.
    """

    def __init__(
        self,
        base_config: dict,
    ):
        self.base_config = base_config

        self.output_dir = (
            Path(
                base_config["project"].get(
                    "output_dir",
                    "./outputs",
                )
            )
            / "tuning"
        )

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

        keys = list(
            search_space.keys()
        )

        values = list(
            search_space.values()
        )

        return [
            dict(zip(keys, combination))
            for combination in itertools.product(
                *values
            )
        ]

    def run_grid_search(
        self,
        search_space: Optional[
            Dict[str, List[Any]]
        ] = None,
        test_ratio: float = 0.15,
        val_ratio: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Execute exhaustive grid search.

        Default search space:

            num_centers:
                [25, 50, 100, 200]

            regularization_lambda:
                [1e-4, 1e-3, 1e-2, 1e-1]

            gamma_multiplier:
                [0.2, 0.5, 1.0, 2.0]

        The total number of configurations is therefore:

            4 x 4 x 4 = 64 trials

        Selection criterion:

            Minimum validation RMSE on the physical reflectance scale.
        """

        # ------------------------------------------------------------------
        # 1. Define default search space
        # ------------------------------------------------------------------
        if search_space is None:
            search_space = {
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

        param_combinations = (
            self._generate_param_grid(
                search_space
            )
        )

        print(
            "Starting RBFN Grid Search "
            f"across {len(param_combinations)} configurations..."
        )

        # ------------------------------------------------------------------
        # 2. Build monthly dataset
        # ------------------------------------------------------------------
        dataset_builder = GapFillDataset(
            self.base_config
        )

        monthly_samples = (
            dataset_builder.build_monthly_samples()
        )

        # ------------------------------------------------------------------
        # 3. Create strict chronological split
        # ------------------------------------------------------------------
        split = (
            dataset_builder.create_temporal_holdout(
                monthly_samples=monthly_samples,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
        )

        print(
            "\nTemporal split:"
        )

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

        # ------------------------------------------------------------------
        # 4. Infer dimensions from actual extracted matrices
        # ------------------------------------------------------------------
        in_features = (
            split.X_train.shape[1]
        )

        out_features = (
            split.Y_train.shape[1]
        )

        # ------------------------------------------------------------------
        # 5. Initialize tuning state
        # ------------------------------------------------------------------
        results: List[
            Dict[str, Any]
        ] = []

        best_val_score = float(
            "inf"
        )

        best_params = None
        best_trainer = None

        # ------------------------------------------------------------------
        # 6. Run grid search
        # ------------------------------------------------------------------
        for idx, params in enumerate(
            param_combinations,
            start=1,
        ):

            # --------------------------------------------------------------
            # Create isolated trial configuration.
            # --------------------------------------------------------------
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

            trial_config["model"][
                "rbfn"
            ].update(params)

            # --------------------------------------------------------------
            # Construct trial trainer
            # --------------------------------------------------------------
            trainer = RBFNTrainer(
                config=trial_config,
                in_features=in_features,
                out_features=out_features,
            )

            # --------------------------------------------------------------
            # Fit scalers ONLY on training data
            # --------------------------------------------------------------
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

            # --------------------------------------------------------------
            # Fit RBFN using training data ONLY
            #
            # K-Means:
            #     training features only
            #
            # Gamma:
            #     derived from training K-Means centres
            #
            # Ridge:
            #     training RBF activations + targets
            # --------------------------------------------------------------
            train_mse_scaled = (
                trainer.fit_ridge(
                    X_train_scaled,
                    Y_train_scaled,
                )
            )

            # --------------------------------------------------------------
            # Evaluate ONLY on validation set for model selection
            # --------------------------------------------------------------
            val_metrics = trainer.evaluate(
                X_eval_scaled=X_val_scaled,
                Y_eval_scaled=Y_val_scaled,
                Y_eval_raw=split.Y_val,
            )

            val_rmse = (
                val_metrics[
                    "eval_rmse_physical"
                ]
            )

            fitted_gamma = float(
                trainer.model.gamma.item()
            )

            # --------------------------------------------------------------
            # Store trial
            # --------------------------------------------------------------
            record = {
                "trial_id": idx,
                "params": params,
                "fitted_gamma": fitted_gamma,
                "train_mse_scaled": train_mse_scaled,
                "val_metrics": val_metrics,
            }

            results.append(
                record
            )

            # --------------------------------------------------------------
            # Console output
            # --------------------------------------------------------------
            print(
                f"[{idx}/{len(param_combinations)}] "
                f"K={params['num_centers']} | "
                f"Lambda={params['regularization_lambda']:.1e} | "
                f"Gamma multiplier={params['gamma_multiplier']:.2f} | "
                f"Gamma={fitted_gamma:.6f} | "
                f"Val RMSE={val_rmse:.6f} | "
                f"Val R²={val_metrics['eval_r2']:.6f}"
            )

            # --------------------------------------------------------------
            # Select best model using validation RMSE
            # --------------------------------------------------------------
            if (
                math.isfinite(val_rmse)
                and val_rmse < best_val_score
            ):
                best_val_score = val_rmse
                best_params = deepcopy(
                    params
                )
                best_trainer = trainer

        # ------------------------------------------------------------------
        # 7. Ensure a valid model was selected
        # ------------------------------------------------------------------
        if best_trainer is None:
            raise RuntimeError(
                "Grid search failed to identify a valid model."
            )

        # ------------------------------------------------------------------
        # 8. Evaluate BEST model on untouched test set
        # ------------------------------------------------------------------
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

        test_metrics = (
            best_trainer.evaluate(
                X_eval_scaled=X_test_scaled,
                Y_eval_scaled=Y_test_scaled,
                Y_eval_raw=split.Y_test,
            )
        )

        final_test_rmse = (
            test_metrics[
                "eval_rmse_physical"
            ]
        )

        final_test_r2 = (
            test_metrics[
                "eval_r2"
            ]
        )

        print(
            f"Final Test RMSE: "
            f"{final_test_rmse:.6f}"
        )

        print(
            f"Final Test R²: "
            f"{final_test_r2:.6f}"
        )

        # ------------------------------------------------------------------
        # 9. Construct tuning summary
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 10. Save tuning history
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 11. Save selected model
        # ------------------------------------------------------------------
        best_trainer.save_checkpoint(
            model_path=str(
                self.output_dir
                / "best_rbfn_model.pt"
            ),
            scaler_path=str(
                self.output_dir
                / "best_rbfn_scalers.pkl"
            ),
            metadata={
                "tuning_summary": summary,
                "selected_for_final_evaluation": True,
                "train_months": split.train_months,
                "val_months": split.val_months,
                "test_months": split.test_months,
            },
        )

        # ------------------------------------------------------------------
        # 12. Final reporting
        # ------------------------------------------------------------------
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
            f"Final test R²: "
            f"{final_test_r2:.6f}"
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