"""Prediction service integration for ETAIQ training workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .drift_detection import DriftDetectionEngine, DriftResult
from .explainability import ExplainabilityEngine, ExplanationResult
from .logging_config import TrainingLogger
from .monitoring import MonitoringEngine, MonitoringRecord
from .prediction_pipeline import PredictionPipelineEngine, PredictionPipelineResult


@dataclass
class PredictionServiceResult:
    """Container for prediction service output and supporting artifacts."""

    prediction: np.ndarray | float | None
    explanation: ExplanationResult | None
    monitoring_record: MonitoringRecord | None
    drift_result: list[DriftResult] | None
    prediction_time: float


class PredictionService:
    """Coordinate prediction, monitoring, drift detection, and explainability."""

    def __init__(
        self,
        logger: TrainingLogger | None = None,
        prediction_pipeline: PredictionPipelineEngine | None = None,
        monitoring_engine: MonitoringEngine | None = None,
        drift_engine: DriftDetectionEngine | None = None,
        explainability_engine: ExplainabilityEngine | None = None,
    ) -> None:
        self._logger = logger or TrainingLogger(name="training.prediction_service")
        self._prediction_pipeline = prediction_pipeline or PredictionPipelineEngine(logger=self._logger)
        self._monitoring_engine = monitoring_engine or MonitoringEngine(logger=self._logger)
        self._drift_engine = drift_engine or DriftDetectionEngine(logger=self._logger)
        self._explainability_engine = explainability_engine or ExplainabilityEngine(logger=self._logger)

    @property
    def monitoring_engine(self) -> MonitoringEngine:
        """Expose the monitoring engine for testing and integration."""
        return self._monitoring_engine

    def set_drift_baseline(self, X_baseline: Any) -> None:
        """Configure a baseline dataset for future drift checks."""
        self._drift_engine.fit_baseline(X_baseline)

    def predict(self, model_path: str | Path, input_data: Any) -> PredictionServiceResult:
        """Run the full prediction workflow for a persisted model."""
        start_time = time.perf_counter()
        self._logger.info("Prediction service started", model_path=str(model_path))

        pipeline_result = self._prediction_pipeline.predict(model_path, input_data)
        self._logger.info("Prediction pipeline completed", rows=pipeline_result.number_of_rows)

        monitoring_record = self._monitoring_engine.record_predictions(
            model_name=pipeline_result.model_name,
            predictions=pipeline_result.predictions,
        )
        self._logger.info("Monitoring recorded", model_name=pipeline_result.model_name)

        drift_result: list[DriftResult] | None = None
        if self._has_drift_baseline():
            drift_result = self._drift_engine.detect_drift(self._prepare_drift_input(input_data))
            self._logger.info("Drift evaluation completed", drift_count=sum(1 for item in drift_result if item.drift_detected))

        explanation = self._generate_explanation(model_path, input_data, pipeline_result)
        self._logger.info("Explanation generated", model_name=pipeline_result.model_name)

        elapsed = time.perf_counter() - start_time
        self._logger.info(
            "Prediction service completed",
            model_name=pipeline_result.model_name,
            prediction_time_seconds=elapsed,
        )

        return PredictionServiceResult(
            prediction=pipeline_result.predictions[0] if len(pipeline_result.predictions) else None,
            explanation=explanation,
            monitoring_record=monitoring_record,
            drift_result=drift_result,
            prediction_time=elapsed,
        )

    def _generate_explanation(self, model_path: str | Path, input_data: Any, pipeline_result: PredictionPipelineResult) -> ExplanationResult | None:
        """Create explanation output for the prediction using the persisted model."""
        model = self._load_model(model_path)
        feature_names = self._collect_feature_names(input_data)
        return self._explainability_engine.explain_model(model, pipeline_result.model_name, feature_names)

    def _load_model(self, model_path: str | Path) -> Any:
        """Load the persisted model from disk for explanation generation."""
        from .persistence import ModelPersistenceEngine

        return ModelPersistenceEngine(logger=self._logger).load_model(model_path)

    def _collect_feature_names(self, input_data: Any) -> list[str]:
        """Extract a stable list of feature names for explanation output."""
        if isinstance(input_data, dict):
            return [str(name) for name in input_data.keys()]
        if isinstance(input_data, (list, tuple)):
            if not input_data:
                return []
            if isinstance(input_data[0], dict):
                return [str(name) for name in input_data[0].keys()]
            return [f"feature_{index}" for index in range(len(input_data[0]))] if input_data and isinstance(input_data[0], (list, tuple, np.ndarray)) else [f"feature_{index}" for index in range(len(input_data))]
        if isinstance(input_data, np.ndarray):
            return [f"feature_{index}" for index in range(input_data.shape[-1] if input_data.ndim > 1 else 1)]
        return [f"feature_{index}" for index in range(1)]

    def _prepare_drift_input(self, input_data: Any) -> Any:
        """Make prediction input suitable for drift analysis."""
        if isinstance(input_data, np.ndarray):
            return input_data.reshape(-1, 1) if input_data.ndim == 1 else input_data
        if isinstance(input_data, (list, tuple)):
            if not input_data:
                return []
            if isinstance(input_data[0], (list, tuple, np.ndarray)):
                return np.asarray(input_data, dtype=float)
            return np.asarray(input_data, dtype=float).reshape(-1, 1)
        return input_data

    def _has_drift_baseline(self) -> bool:
        """Check whether drift detection has a fitted baseline."""
        return self._drift_engine._baseline is not None
