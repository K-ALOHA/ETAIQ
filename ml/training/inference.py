"""Inference engine for the ETAIQ production training pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .logging_config import TrainingLogger
from .persistence import ModelPersistenceEngine


@dataclass
class InferenceResult:
    """Container for a single prediction request."""

    model_name: str
    prediction: Any
    prediction_time_seconds: float
    input_rows: int
    model_version: int


class InferenceEngine:
    """Load persisted models and produce predictions for single or batched input."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.inference")
        self._persistence = ModelPersistenceEngine(logger=logger)

    def predict(self, model_path: str | Path, X: Any, model: Any | None = None) -> InferenceResult:
        """Load a saved model and return a single prediction result."""
        model = self._load_model(model_path) if model is None else model
        features = self._prepare_features(X)

        self._validate_features(features)
        features = self._align_features_to_model(features, model)
        self._logger.info("Prediction started", model_path=str(model_path), rows=len(features))

        start_time = time.perf_counter()
        prediction = model.predict(features)
        elapsed = time.perf_counter() - start_time

        self._logger.info(
            "Prediction completed",
            model_path=str(model_path),
            rows=len(features),
            prediction_time_seconds=elapsed,
        )

        return InferenceResult(
            model_name=self._extract_model_name(model_path),
            prediction=np.asarray(prediction).reshape(-1)[0],
            prediction_time_seconds=elapsed,
            input_rows=len(features),
            model_version=self._extract_model_version(model_path),
        )

    def batch_predict(self, model_path: str | Path, X: Any, model: Any | None = None) -> np.ndarray:
        """Load a saved model and return batched predictions."""
        model = self._load_model(model_path) if model is None else model
        features = self._prepare_features(X)

        self._validate_features(features)
        features = self._align_features_to_model(features, model)
        self._logger.info("Prediction started", model_path=str(model_path), rows=len(features))

        start_time = time.perf_counter()
        predictions = model.predict(features)
        elapsed = time.perf_counter() - start_time

        self._logger.info(
            "Prediction completed",
            model_path=str(model_path),
            rows=len(features),
            prediction_time_seconds=elapsed,
        )

        return np.asarray(predictions).reshape(-1)

    def _load_model(self, model_path: str | Path) -> Any:
        """Load a persisted model via the persistence engine."""
        path = Path(model_path)
        if not path.exists():
            raise ValueError(f"Model file missing: {path}")
        self._logger.info("Model loading", model_path=str(path))
        return self._persistence.load_model(path)

    def _prepare_features(self, X: Any) -> np.ndarray:
        """Convert dataframe or array input into a numeric ndarray."""
        if X is None:
            raise ValueError("Input data cannot be None")

        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Input data cannot be empty")
            return X.to_numpy()

        if isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("Input data cannot be empty")
            return np.asarray(X)

        if isinstance(X, list):
            if len(X) == 0:
                raise ValueError("Input data cannot be empty")
            return np.asarray(X)

        raise ValueError("Unsupported input type")

    def _validate_features(self, features: np.ndarray) -> None:
        """Ensure the feature matrix is not empty and has a valid shape."""
        if features.size == 0:
            raise ValueError("Input data cannot be empty")

        if features.ndim == 0:
            raise ValueError("Input data must be at least 1-dimensional")

        if features.ndim == 1:
            features = features.reshape(1, -1)

    def _align_features_to_model(self, features: np.ndarray, model: Any) -> np.ndarray:
        """Pad or truncate feature matrices so they match the model's expected input width."""
        expected_count = self._extract_expected_feature_count(model)
        if expected_count is None:
            return features

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] == expected_count:
            return features

        if features.shape[1] < expected_count:
            pad_width = expected_count - features.shape[1]
            return np.pad(features, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

        return features[:, :expected_count]

    def _extract_expected_feature_count(self, model: Any) -> int | None:
        """Inspect a fitted model for the expected number of input features."""
        if model is None:
            return None
        if hasattr(model, "n_features_in_"):
            return int(getattr(model, "n_features_in_"))
        if hasattr(model, "named_steps"):
            for _, step in reversed(list(model.named_steps.items())):
                if hasattr(step, "n_features_in_"):
                    return int(getattr(step, "n_features_in_"))
        return None

    def _extract_model_name(self, model_path: str | Path) -> str:
        """Infer the model name from the persisted file path."""
        path = Path(model_path)
        return path.stem.split("_v")[0]

    def _extract_model_version(self, model_path: str | Path) -> int:
        """Infer the model version from the persisted file path."""
        path = Path(model_path)
        suffix = path.stem.split("_v", maxsplit=1)[1]
        return int(suffix)
