"""Training engine for the ETAIQ production training pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .logging_config import TrainingLogger
from .registry import ModelRegistry


@dataclass
class TrainingResult:
    """Container for a completed training run."""

    model_name: str
    trained_model: Any
    training_time_seconds: float
    training_rows: int
    training_features: int


class TrainingEngine:
    """Focused training engine for fitting registered regression models."""

    def __init__(self, registry: ModelRegistry | None = None, logger: TrainingLogger | None = None) -> None:
        self._registry = registry or ModelRegistry()
        self._logger = logger or TrainingLogger(name="training.engine")

    def train(self, model_name: str, X_train: Any, y_train: Any) -> TrainingResult:
        """Validate the data, fit a fresh model, and return training metadata."""
        self._validate_inputs(model_name, X_train, y_train)

        self._logger.info(
            "Starting training",
            model_name=model_name,
            training_rows=len(np.asarray(X_train)),
            training_features=np.asarray(X_train).shape[1] if np.asarray(X_train).ndim > 1 else 1,
        )

        start_time = time.perf_counter()
        model = self._registry.get_model(model_name)
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start_time

        self._logger.info(
            "Training completed",
            model_name=model_name,
            training_time_seconds=elapsed,
            training_rows=len(np.asarray(X_train)),
            training_features=np.asarray(X_train).shape[1] if np.asarray(X_train).ndim > 1 else 1,
        )

        return TrainingResult(
            model_name=model_name,
            trained_model=model,
            training_time_seconds=elapsed,
            training_rows=len(np.asarray(X_train)),
            training_features=np.asarray(X_train).shape[1] if np.asarray(X_train).ndim > 1 else 1,
        )

    def _validate_inputs(self, model_name: str, X_train: Any, y_train: Any) -> None:
        """Validate the model name and dataset shapes before training."""
        if not self._registry.has_model(model_name):
            raise ValueError(f"Unknown model name: {model_name}")

        X_array = np.asarray(X_train)
        y_array = np.asarray(y_train)

        if X_array.size == 0 or y_array.size == 0:
            raise ValueError("Training data cannot be empty")

        if len(X_array) != len(y_array):
            raise ValueError("Training features and targets must have the same length")
