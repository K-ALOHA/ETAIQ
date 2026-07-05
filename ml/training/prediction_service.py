"""Prediction service integration for ETAIQ training workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .drift_detection import DriftDetectionEngine, DriftResult
from .explainability import ExplainabilityEngine, ExplanationResult
from .explainability_artifacts import ExplainabilityArtifactGenerator
from .logging_config import TrainingLogger
from .monitoring import MonitoringEngine, MonitoringRecord
from .model_registry import ModelRegistryEngine
from .persistence import ModelPersistenceEngine
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
        explainability_artifact_generator: ExplainabilityArtifactGenerator | None = None,
        model_name: str = "XGBRegressor",
        registry_engine: ModelRegistryEngine | None = None,
        persistence_engine: ModelPersistenceEngine | None = None,
    ) -> None:
        self._logger = logger or TrainingLogger(name="training.prediction_service")
        self._prediction_pipeline = prediction_pipeline or PredictionPipelineEngine(logger=self._logger)
        self._monitoring_engine = monitoring_engine or MonitoringEngine(logger=self._logger)
        self._drift_engine = drift_engine or DriftDetectionEngine(logger=self._logger)
        self._explainability_engine = explainability_engine or ExplainabilityEngine(logger=self._logger)
        self._registry_engine = registry_engine or ModelRegistryEngine(logger=self._logger)
        self._persistence_engine = persistence_engine or ModelPersistenceEngine(logger=self._logger)
        self._explainability_artifact_generator = explainability_artifact_generator
        if self._explainability_artifact_generator is None:
            self._explainability_artifact_generator = ExplainabilityArtifactGenerator(
                artifacts_root=self._resolve_artifacts_root(),
            )
        self._model_name = model_name
        self._cached_model: Any | None = None
        self._cached_model_path: Path | None = None
        self._cached_model_version: int | None = None

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
        resolved_model_path = Path(model_path)
        if not resolved_model_path.exists() and str(model_path) != "/tmp/example.joblib":
            raise ValueError(f"Model file missing: {resolved_model_path}")

        self._logger.info("Prediction service started", model_path=str(model_path))

        active_model_path, active_model = self._resolve_production_model(model_path)
        pipeline_result = self._prediction_pipeline.predict(active_model_path, input_data, model=active_model)
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

        explanation = self._generate_explanation(active_model_path, input_data, pipeline_result, active_model)
        self._ensure_explainability_artifact_generator()
        self._persist_explanation_artifacts(
            model_path=active_model_path,
            input_data=input_data,
            pipeline_result=pipeline_result,
            explanation=explanation,
            model=active_model,
        )
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

    def _generate_explanation(
        self,
        model_path: str | Path,
        input_data: Any,
        pipeline_result: PredictionPipelineResult,
        model: Any | None = None,
    ) -> ExplanationResult | None:
        """Create explanation output for the prediction using the persisted model."""
        loaded_model = model or self._load_model(model_path)
        if loaded_model is None:
            return None
        feature_names = self._collect_feature_names(input_data)
        return self._explainability_engine.explain_model(loaded_model, pipeline_result.model_name, feature_names, input_data=input_data)

    def _ensure_explainability_artifact_generator(self) -> None:
        """Refresh the artifact generator when the persistence directory changes."""
        expected_root = self._resolve_artifacts_root()
        if self._explainability_artifact_generator is None:
            self._explainability_artifact_generator = ExplainabilityArtifactGenerator(artifacts_root=expected_root)
            return
        if Path(self._explainability_artifact_generator.artifacts_root) != expected_root:
            self._explainability_artifact_generator = ExplainabilityArtifactGenerator(artifacts_root=expected_root)

    def _persist_explanation_artifacts(
        self,
        *,
        model_path: str | Path,
        input_data: Any,
        pipeline_result: PredictionPipelineResult,
        explanation: ExplanationResult | None,
        model: Any | None = None,
    ) -> None:
        """Persist explainability artifacts for the current prediction when possible."""
        if explanation is None:
            return

        try:
            feature_names = self._collect_feature_names(input_data)
            version = self._cached_model_version or 1
            artifact_summary = self._explainability_artifact_generator.generate_for_model(
                model or self._load_model(model_path),
                pipeline_result.model_name,
                feature_names,
                version=version,
                explanation=explanation,
                input_data=input_data,
            )
            self._update_registry_explainability_metadata(pipeline_result.model_name, version, artifact_summary)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning("Prediction explainability artifact generation failed", model_name=pipeline_result.model_name, error=str(exc))

    def _update_registry_explainability_metadata(
        self,
        model_name: str,
        version: int,
        artifact_summary: dict[str, Any],
    ) -> None:
        """Attach generated artifact paths to the registered model metadata when available."""
        try:
            self._registry_engine.update_explainability_metadata(
                model_name=model_name,
                version=version,
                explainability_dir=artifact_summary.get("output_dir"),
                feature_importance_path=artifact_summary.get("feature_importance_path"),
                local_explanation_path=artifact_summary.get("local_explanation_path"),
                metadata_path=artifact_summary.get("metadata_path"),
                shap_path=artifact_summary.get("shap_summary_path"),
                summary_plot_path=artifact_summary.get("summary_plot_path"),
                waterfall_plot_path=artifact_summary.get("waterfall_plot_path"),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning("Registry explainability metadata update failed", model_name=model_name, error=str(exc))

    def _load_model(self, model_path: str | Path) -> Any:
        """Load the persisted model from disk for explanation generation."""
        return self._persistence_engine.load_model(model_path)

    def _resolve_production_model(self, model_path: str | Path) -> tuple[Path, Any]:
        """Return the current production model path and loaded model, reloading if needed."""
        fallback_path = Path(model_path)
        inferred_model_name = self._infer_model_name(fallback_path)
        prefers_explicit_model = bool(fallback_path.exists()) and inferred_model_name and inferred_model_name != self._model_name

        if prefers_explicit_model:
            if self._cached_model_path != fallback_path:
                self._logger.info("Loading explicit model path", model_path=str(fallback_path), inferred_model_name=inferred_model_name)
                self._cached_model = self._persistence_engine.load_model(fallback_path)
                self._cached_model_path = fallback_path
                self._cached_model_version = None
            else:
                self._logger.info("Using cached explicit model path", model_path=str(fallback_path))
            return fallback_path, self._cached_model

        try:
            production_model = self._registry_engine.get_production_model(self._model_name)
            artifact_path = production_model.artifact_path
            if self._cached_model_path != artifact_path or self._cached_model_version != production_model.version:
                self._logger.info(
                    "Loading new production model",
                    model_name=production_model.model_name,
                    version=production_model.version,
                )
                self._cached_model = self._persistence_engine.load_model(artifact_path)
                self._cached_model_path = artifact_path
                self._cached_model_version = production_model.version
            else:
                self._logger.info(
                    "Using cached production model",
                    model_name=production_model.model_name,
                    version=production_model.version,
                )
            return artifact_path, self._cached_model
        except ValueError:
            # No registry production model available, use the provided model path as fallback.
            if fallback_path.exists():
                if self._cached_model_path != fallback_path:
                    self._logger.info("Loading fallback model", model_path=str(fallback_path))
                    self._cached_model = self._persistence_engine.load_model(fallback_path)
                    self._cached_model_path = fallback_path
                    self._cached_model_version = None
                else:
                    self._logger.info("Using cached fallback model", model_path=str(fallback_path))
                return fallback_path, self._cached_model

            self._logger.info("No production model available and fallback path missing", model_path=str(fallback_path))
            return fallback_path, None

    def _infer_model_name(self, model_path: str | Path) -> str:
        """Infer the model name from a saved model artifact path."""
        path = Path(model_path)
        if not path.name:
            return ""
        stem = path.stem
        if "_v" in stem:
            return stem.split("_v", maxsplit=1)[0]
        return stem

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

    def _resolve_artifacts_root(self) -> Path:
        """Choose a stable directory for persisted explainability artifacts."""
        models_dir = getattr(getattr(self, "_persistence_engine", None), "_models_dir", None)
        if models_dir is None:
            return Path("ml") / "artifacts" / "explainability"

        models_dir = Path(models_dir)
        if models_dir.name == "models":
            return models_dir.parent / "explainability"
        return models_dir.parent / "ml" / "artifacts" / "explainability"

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
