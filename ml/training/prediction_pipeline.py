"""End-to-end prediction pipeline for persisted ETAIQ regression models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ml.features.encoding import EncodingEngine
from ml.features.encoding_registry import EncodingPlan, EncodingPlanEntry
from ml.features.scaling import ScalingEngine
from ml.features.selection import FeatureSelectionEngine

from .inference import InferenceEngine
from .logging_config import TrainingLogger
from .persistence import ModelPersistenceEngine


@dataclass
class PredictionPipelineResult:
    """Container for predictions produced by the end-to-end prediction pipeline."""

    predictions: np.ndarray
    number_of_rows: int
    model_name: str
    model_version: int
    pipeline_time_seconds: float


class PredictionPipelineEngine:
    """Transform raw features and score them with a persisted model."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.prediction_pipeline")
        self._encoding_engine = EncodingEngine(logger=None)
        self._scaling_engine = ScalingEngine(logger=None)
        self._selection_engine = FeatureSelectionEngine(logger=None)
        self._inference_engine = InferenceEngine(logger=logger)
        self._persistence_engine = ModelPersistenceEngine(logger=logger)

    def predict(self, model_path: str | Path, X: Any) -> PredictionPipelineResult:
        """Prepare raw features, load a persisted model, and return predictions."""
        self._validate_inputs(model_path, X)

        self._logger.info("Pipeline started", model_path=str(model_path))
        start_time = time.perf_counter()

        features_frame = self._prepare_dataframe(X)
        self._logger.info("Encoding completed")
        encoded_frame = self._encode_features(features_frame)
        self._logger.info("Scaling completed")
        scaled_frame = self._scale_features(encoded_frame)
        self._logger.info("Feature selection completed")
        selected_frame = self._select_features(scaled_frame)

        self._logger.info("Model loaded", model_path=str(model_path))
        predictions = self._inference_engine.batch_predict(model_path, selected_frame)
        if predictions.ndim != 1:
            predictions = np.asarray(predictions).reshape(-1)

        elapsed = time.perf_counter() - start_time
        self._logger.info("Prediction completed", rows=len(selected_frame), pipeline_time_seconds=elapsed)
        self._logger.info("Pipeline completed", pipeline_time_seconds=elapsed)

        return PredictionPipelineResult(
            predictions=predictions,
            number_of_rows=len(selected_frame),
            model_name=self._inference_engine._extract_model_name(model_path),
            model_version=self._inference_engine._extract_model_version(model_path),
            pipeline_time_seconds=elapsed,
        )

    def _validate_inputs(self, model_path: str | Path, X: Any) -> None:
        """Validate model path and raw input data."""
        if X is None:
            raise ValueError("Input data cannot be None")

        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Input data cannot be empty")
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("Input data cannot be empty")
        elif isinstance(X, list):
            if len(X) == 0:
                raise ValueError("Input data cannot be empty")
        else:
            raise ValueError("Unsupported input type")

        path = Path(model_path)
        if not path.exists():
            raise ValueError(f"Model file missing: {path}")

    def _prepare_dataframe(self, X: Any) -> pd.DataFrame:
        """Convert raw input into a pandas DataFrame."""
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        elif isinstance(X, np.ndarray):
            array = np.asarray(X)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            frame = pd.DataFrame(array)
        else:
            frame = pd.DataFrame(X)

        if frame.empty:
            raise ValueError("Input data cannot be empty")

        frame.columns = [str(column) for column in frame.columns]
        return frame

    def _encode_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply a minimal encoding plan to the raw feature dataframe."""
        plan = EncodingPlan()
        for column in X.columns:
            feature_type = "Numerical" if np.issubdtype(X[column].dtype, np.number) else "Categorical"
            plan.add_entry(str(column), feature_type, "No Encoding")

        self._encoding_engine.fit(X, plan=plan)
        encoded_train, encoded_test = self._encoding_engine.transform(X, X)
        return encoded_train

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature scaling to the encoded dataframe."""
        plan_path = Path("ml") / "data" / "features" / "encoding_plan.csv"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan = EncodingPlan()
        for column in X.columns:
            plan.add_entry(str(column), "Numerical", "No Encoding")
        with plan_path.open("w", encoding="utf-8") as handle:
            handle.write("feature_name,feature_type,encoding_strategy\n")
            for entry in plan.entries:
                handle.write(f"{entry.feature_name},{entry.feature_type},{entry.encoding_strategy}\n")

        self._scaling_engine.fit(X, plan_path=plan_path)
        scaled_train, _ = self._scaling_engine.transform(X, X)
        return scaled_train

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select a stable subset of features using the feature selection engine."""
        y = pd.Series(np.arange(len(X), dtype=float))
        selected_train, _ = self._selection_engine.select_features(X, X, y)
        return selected_train
