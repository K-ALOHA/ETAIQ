"""Training engine for the ETAIQ production training pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .logging_config import TrainingLogger
from .registry import ModelRegistry
from ml.features.encoding import EncodingEngine
from ml.features.encoding_registry import EncodingPlan
from ml.features.scaling import ScalingEngine
from ml.features.selection import FeatureSelectionEngine
from ml.features.sklearn_preprocessor import SklearnPreprocessor
from sklearn.pipeline import Pipeline


@dataclass
class TrainingResult:
    """Container for a completed training run."""

    model_name: str
    trained_model: Any
    training_time_seconds: float
    training_rows: int
    training_features: int
    selected_features: list[str] | None = None


class TrainingEngine:
    """Focused training engine for fitting registered regression models."""

    def __init__(self, registry: ModelRegistry | None = None, logger: TrainingLogger | None = None) -> None:
        self._registry = registry or ModelRegistry()
        self._logger = logger or TrainingLogger(name="training.engine")
        self._encoding_engine = EncodingEngine(logger=None)
        self._scaling_engine = ScalingEngine(logger=None)
        self._selection_engine = FeatureSelectionEngine(logger=None)

    def train(self, model_name: str, X_train: Any, y_train: Any) -> TrainingResult:
        """Validate the data, fit a fresh model, and return training metadata."""
        self._validate_inputs(model_name, X_train, y_train)

        self._logger.info(
            "Starting training",
            model_name=model_name,
            training_rows=len(np.asarray(X_train)),
            training_features=np.asarray(X_train).shape[1] if np.asarray(X_train).ndim > 1 else 1,
        )

        # Preprocess and train using a sklearn Pipeline (preprocessor + estimator)
        import pandas as pd
        start_time = time.perf_counter()

        preprocessor = SklearnPreprocessor()
        estimator = self._registry.get_model(model_name)
        pipeline = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])

        # Fit pipeline directly on raw inputs so preprocessing is consistently applied
        pipeline.fit(X_train, y_train)
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
            trained_model=pipeline,
            training_time_seconds=elapsed,
            training_rows=len(np.asarray(X_train)),
            training_features=np.asarray(X_train).shape[1] if np.asarray(X_train).ndim > 1 else 1,
            selected_features=getattr(preprocessor, "selected_features_", None),
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
