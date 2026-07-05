"""Evaluation engine for the ETAIQ production training pipeline."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import DEFAULT_TRAINING_CONFIG
from .logging_config import TrainingLogger


@dataclass
class EvaluationResult:
    """Container for a completed model evaluation run."""

    model_name: str
    predictions: Any
    mae: float
    rmse: float
    r2: float
    mape: float
    evaluation_time_seconds: float
    evaluation_rows: int


class EvaluationEngine:
    """Focused evaluation engine for regression models."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.evaluation")
        self._export_path = DEFAULT_TRAINING_CONFIG.models_dir / "evaluation_metrics.csv"

    def evaluate(self, model: Any, model_name: str, X_test: Any, y_test: Any) -> EvaluationResult:
        """Validate inputs, score a model, and return evaluation metrics."""
        self._validate_inputs(model, model_name, X_test, y_test)

        X_array = np.asarray(X_test)
        y_array = np.asarray(y_test)

        self._logger.info(
            "Evaluation started",
            model_name=model_name,
            rows=len(X_array),
        )

        start_time = time.perf_counter()
        # Prefer passing the original X_test (DataFrame or array) to preserve column names
        if hasattr(model, "predict"):
            try:
                predictions = model.predict(X_test)
            except Exception:
                # fallback to numpy array if the model requires raw arrays
                predictions = model.predict(X_array)
        elif callable(model):
            try:
                predictions = np.asarray(model(X_test))
            except Exception:
                predictions = np.asarray(model(X_array))
        else:
            raise ValueError("Model must be callable or expose a predict method")
        elapsed = time.perf_counter() - start_time

        predictions = np.asarray(predictions).reshape(-1)
        y_array = np.asarray(y_array).reshape(-1)

        mae = float(mean_absolute_error(y_array, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_array, predictions)))
        r2 = float(r2_score(y_array, predictions))
        mape = float(np.mean(np.abs((y_array - predictions) / np.maximum(np.abs(y_array), 1e-8))) * 100.0)

        self._logger.info(
            "Evaluation completed",
            model_name=model_name,
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            evaluation_time_seconds=elapsed,
            rows=len(X_array),
        )

        return EvaluationResult(
            model_name=model_name,
            predictions=predictions,
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            evaluation_time_seconds=elapsed,
            evaluation_rows=len(X_array),
        )

    def export_metrics(self, result: EvaluationResult) -> Path:
        """Persist evaluation metrics to a CSV file."""
        self._export_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "model_name",
            "mae",
            "rmse",
            "r2",
            "mape",
            "evaluation_time_seconds",
            "evaluation_rows",
        ]

        file_exists = self._export_path.exists()
        with self._export_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "model_name": result.model_name,
                    "mae": result.mae,
                    "rmse": result.rmse,
                    "r2": result.r2,
                    "mape": result.mape,
                    "evaluation_time_seconds": result.evaluation_time_seconds,
                    "evaluation_rows": result.evaluation_rows,
                }
            )

        return self._export_path

    def _validate_inputs(self, model: Any, model_name: str, X_test: Any, y_test: Any) -> None:
        """Validate that evaluation inputs are present and aligned."""
        if model is None:
            raise ValueError("Model cannot be None")

        X_array = np.asarray(X_test)
        y_array = np.asarray(y_test)

        if X_array.size == 0 or y_array.size == 0:
            raise ValueError("Evaluation data cannot be empty")

        if len(X_array) != len(y_array):
            raise ValueError("Evaluation features and targets must have the same length")

        if not model_name:
            raise ValueError("Model name cannot be empty")
