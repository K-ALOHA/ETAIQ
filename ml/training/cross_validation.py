"""Cross-validation engine for the ETAIQ production training pipeline."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from .config import DEFAULT_TRAINING_CONFIG
from .evaluation import EvaluationEngine, EvaluationResult
from .logging_config import TrainingLogger
from .registry import ModelRegistry
from .trainer import TrainingEngine


@dataclass
class CrossValidationResult:
    """Container for a completed cross-validation run."""

    model_name: str
    fold_results: list[EvaluationResult]
    mean_mae: float
    std_mae: float
    mean_rmse: float
    std_rmse: float
    mean_r2: float
    std_r2: float
    mean_mape: float
    std_mape: float
    total_training_time_seconds: float
    number_of_folds: int


class CrossValidationEngine:
    """Cross-validation engine that reuses the training and evaluation engines."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.cross_validation")
        self._registry = ModelRegistry()
        self._trainer = TrainingEngine(registry=self._registry)
        self._evaluator = EvaluationEngine()
        self._export_path = DEFAULT_TRAINING_CONFIG.models_dir / "cross_validation_results.csv"

    def cross_validate(
        self,
        model_name: str,
        X: Any,
        y: Any,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> CrossValidationResult:
        """Run K-fold cross-validation for the specified model and dataset."""
        self._validate_inputs(model_name, X, y, n_splits)

        X_array = np.asarray(X)
        y_array = np.asarray(y)
        kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        self._logger.info(
            "Cross validation started",
            model_name=model_name,
            number_of_folds=n_splits,
        )

        fold_results: list[EvaluationResult] = []
        total_training_time = 0.0
        start_time = time.perf_counter()

        for fold_number, (train_idx, test_idx) in enumerate(kfold.split(X_array), start=1):
            self._logger.info("Fold start", model_name=model_name, fold=fold_number)

            fold_training_start = time.perf_counter()
            training_result = self._trainer.train(model_name, X_array[train_idx], y_array[train_idx])
            total_training_time += time.perf_counter() - fold_training_start

            evaluation_result = self._evaluator.evaluate(
                training_result.trained_model,
                model_name,
                X_array[test_idx],
                y_array[test_idx],
            )
            fold_results.append(evaluation_result)
            self._logger.info(
                "Fold completion",
                model_name=model_name,
                fold=fold_number,
                mae=evaluation_result.mae,
                rmse=evaluation_result.rmse,
                r2=evaluation_result.r2,
                mape=evaluation_result.mape,
            )

        elapsed = time.perf_counter() - start_time
        aggregated = self._aggregate_results(fold_results)
        self._logger.info(
            "Aggregated metrics",
            model_name=model_name,
            mean_mae=aggregated["mean_mae"],
            mean_rmse=aggregated["mean_rmse"],
            mean_r2=aggregated["mean_r2"],
            mean_mape=aggregated["mean_mape"],
            total_training_time_seconds=elapsed,
        )
        self._logger.info(
            "Cross validation completed",
            model_name=model_name,
            completion_time_seconds=elapsed,
        )

        return CrossValidationResult(
            model_name=model_name,
            fold_results=fold_results,
            mean_mae=aggregated["mean_mae"],
            std_mae=aggregated["std_mae"],
            mean_rmse=aggregated["mean_rmse"],
            std_rmse=aggregated["std_rmse"],
            mean_r2=aggregated["mean_r2"],
            std_r2=aggregated["std_r2"],
            mean_mape=aggregated["mean_mape"],
            std_mape=aggregated["std_mape"],
            total_training_time_seconds=total_training_time,
            number_of_folds=n_splits,
        )

    def export_results(self, result: CrossValidationResult) -> Path:
        """Persist cross-validation results to a CSV file."""
        self._export_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["fold", "mae", "rmse", "r2", "mape"]
        file_exists = self._export_path.exists()
        with self._export_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for index, fold_result in enumerate(result.fold_results, start=1):
                writer.writerow(
                    {
                        "fold": index,
                        "mae": fold_result.mae,
                        "rmse": fold_result.rmse,
                        "r2": fold_result.r2,
                        "mape": fold_result.mape,
                    }
                )

        return self._export_path

    def _aggregate_results(self, fold_results: list[EvaluationResult]) -> dict[str, float]:
        """Aggregate fold-level metrics into mean and standard deviation values."""
        mae_values = [result.mae for result in fold_results]
        rmse_values = [result.rmse for result in fold_results]
        r2_values = [result.r2 for result in fold_results]
        mape_values = [result.mape for result in fold_results]

        return {
            "mean_mae": float(np.mean(mae_values)),
            "std_mae": float(np.std(mae_values)),
            "mean_rmse": float(np.mean(rmse_values)),
            "std_rmse": float(np.std(rmse_values)),
            "mean_r2": float(np.mean(r2_values)),
            "std_r2": float(np.std(r2_values)),
            "mean_mape": float(np.mean(mape_values)),
            "std_mape": float(np.std(mape_values)),
        }

    def _validate_inputs(self, model_name: str, X: Any, y: Any, n_splits: int) -> None:
        """Validate model name, dataset shape, and fold count."""
        if not self._registry.has_model(model_name):
            raise ValueError(f"Unknown model name: {model_name}")

        X_array = np.asarray(X)
        y_array = np.asarray(y)

        if X_array.size == 0 or y_array.size == 0:
            raise ValueError("Training data cannot be empty")

        if len(X_array) != len(y_array):
            raise ValueError("Training features and targets must have the same length")

        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        if n_splits > len(X_array):
            raise ValueError("n_splits cannot exceed the number of rows")
