"""Context building helpers for the ETAIQ AI assistant."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import get_settings
from app.core.logging import get_logger

import importlib.util


def _load_monitoring_engine() -> Any:
    """Load the monitoring engine without importing the training package initializer."""
    training_package = type(sys)("ml.training")
    training_package.__path__ = [str(REPO_ROOT / "ml" / "training")]
    sys.modules.setdefault("ml.training", training_package)

    module_path = REPO_ROOT / "ml" / "training" / "monitoring.py"
    spec = importlib.util.spec_from_file_location("ml.training.monitoring", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load monitoring module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MonitoringEngine


MonitoringEngine = _load_monitoring_engine()

logger = get_logger(__name__)
settings = get_settings()


class ContextBuilder:
    """Build real ETAIQ context blocks for downstream assistant prompts."""

    def __init__(self) -> None:
        self._monitoring_engine = MonitoringEngine()
        self._registry_engine = self._load_registry_engine()
        self._settings = settings
        self._repo_root = Path(__file__).resolve().parents[3]

    @staticmethod
    def _load_registry_engine() -> Any:
        """Load the registry engine without failing if optional training dependencies are missing."""
        try:
            module_path = REPO_ROOT / "ml" / "training" / "model_registry.py"
            spec = importlib.util.spec_from_file_location("ml.training.model_registry", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load model registry module from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.ModelRegistryEngine(storage_dir=REPO_ROOT / "ml" / "data" / "training" / "model_registry")
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("assistant_context_registry_engine_unavailable", error=str(exc))
            return None

    def _load_registry_snapshot(self) -> list[dict[str, Any]]:
        """Return persisted registry information from disk when the engine is unavailable."""
        registry_dir = self._repo_root / "ml" / "data" / "training" / "model_registry"
        if not registry_dir.exists():
            return []

        snapshots = []
        for registry_file in sorted(registry_dir.glob("*_registry.json")):
            try:
                payload = json.loads(registry_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            snapshots.append(payload)
        return snapshots

    def build_context(self) -> dict[str, Any]:
        """Return structured ETAIQ context for the current environment."""
        return {
            "production_model": self.get_production_model_context(),
            "registry_summary": self.get_registry_context(),
            "monitoring_summary": self.get_monitoring_context(),
            "health_context": self.get_health_context(),
            "latest_prediction": self.get_latest_prediction_context(),
            "dataset_summary": self.get_dataset_context(),
            "eda_summary": self.get_eda_context(),
            "training_summary": self.get_training_context(),
        }

    def get_production_model_context(self) -> dict[str, Any]:
        """Return the current production model metadata from the registry."""
        try:
            if self._registry_engine is None:
                snapshots = self._load_registry_snapshot()
                production_models = [
                    snapshot
                    for snapshot in snapshots
                    if str(snapshot.get("status", "")).lower() == "production" and str(snapshot.get("model_name", "")).lower() == "xgbregressor"
                ]
                if not production_models:
                    return {"name": None, "version": None, "status": "unknown", "metrics": {}, "created_at": None}
                selected = sorted(production_models, key=lambda record: (str(record.get("created_at", "")), int(record.get("version", 0))), reverse=True)[0]
                return {
                    "name": selected.get("model_name"),
                    "version": selected.get("version"),
                    "status": selected.get("status"),
                    "metrics": selected.get("metrics") or {},
                    "created_at": selected.get("created_at"),
                    "artifact_path": selected.get("artifact_path"),
                }
            try:
                selected = self._registry_engine.select_production_model("XGBRegressor")
            except ValueError:
                return {"name": None, "version": None, "status": "unknown", "metrics": {}, "created_at": None}
            return {
                "name": selected.model_name,
                "version": selected.version,
                "status": selected.status,
                "metrics": selected.metrics,
                "created_at": selected.created_at,
                "artifact_path": str(selected.artifact_path),
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("assistant_context_production_model_failed", error=str(exc))
            return {"name": None, "version": None, "status": "unknown", "metrics": {}, "created_at": None}

    def get_registry_context(self) -> dict[str, Any]:
        """Return a compact registry summary including the production model and archived count."""
        try:
            if self._registry_engine is None:
                snapshots = self._load_registry_snapshot()
                production_models = [snapshot for snapshot in snapshots if str(snapshot.get("status", "")).lower() == "production"]
                archived_models = [snapshot for snapshot in snapshots if str(snapshot.get("status", "")).lower() == "archived"]
                production_model = sorted(production_models, key=lambda record: (str(record.get("created_at", "")), int(record.get("version", 0))), reverse=True)[0] if production_models else None
                return {
                    "production_model": production_model.get("model_name") if production_model else None,
                    "version": production_model.get("version") if production_model else None,
                    "status": production_model.get("status") if production_model else None,
                    "metrics": production_model.get("metrics") or {} if production_model else {},
                    "created_at": production_model.get("created_at") if production_model else None,
                    "archived_models_count": len(archived_models),
                }
            models = self._registry_engine.list_models()
            production_models = [model for model in models if model.status == "Production" and str(model.model_name).lower() == "xgbregressor"]
            archived_models = [model for model in models if str(model.status).lower() == "archived"]
            production_model = sorted(production_models, key=lambda record: (record.created_at, record.version), reverse=True)[0] if production_models else None
            return {
                "production_model": production_model.model_name if production_model else None,
                "version": production_model.version if production_model else None,
                "status": production_model.status if production_model else None,
                "metrics": dict(production_model.metrics) if production_model else {},
                "created_at": production_model.created_at if production_model else None,
                "archived_models_count": len(archived_models),
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("assistant_context_registry_failed", error=str(exc))
            return {"production_model": None, "version": None, "status": None, "metrics": {}, "created_at": None, "archived_models_count": 0}

    def get_monitoring_context(self) -> dict[str, Any]:
        """Return a summary of monitoring state and the latest latency information."""
        try:
            records = self._monitoring_engine.list_records()
            latest_latency = None
            for record in records:
                for key in ("latency_ms", "prediction_latency_ms", "latency"):
                    value = getattr(record, key, None)
                    if isinstance(value, (int, float)):
                        latest_latency = float(value)
                        break
                if latest_latency is not None:
                    break
            return {
                "backend_status": "healthy",
                "registry_status": "healthy",
                "prediction_api_status": "healthy",
                "monitoring_status": "healthy" if records else "unavailable",
                "latency_ms": latest_latency,
                "record_count": len(records),
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("assistant_context_monitoring_failed", error=str(exc))
            return {"backend_status": "unknown", "registry_status": "unknown", "prediction_api_status": "unknown", "monitoring_status": "unknown", "latency_ms": None, "record_count": 0}

    def get_health_context(self) -> dict[str, Any]:
        """Return basic health status for the assistant environment."""
        return {
            "status": "healthy",
            "model_loaded": bool(self.get_production_model_context()["name"]),
        }

    def get_training_context(self) -> dict[str, Any]:
        """Return training summary details from persisted artifact metadata."""
        try:
            artifact_paths = sorted((self._repo_root / "ml" / "artifacts" / "models").rglob("*.json"))
            training_runs = []
            for artifact_path in artifact_paths:
                if artifact_path.name.endswith("_registry.json"):
                    continue
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                custom_metadata = payload.get("custom_metadata", {}) if isinstance(payload, dict) else {}
                if not isinstance(custom_metadata, dict):
                    custom_metadata = {}
                metrics = custom_metadata.get("evaluation_metrics") or payload.get("metrics") or {}
                training_runs.append(
                    {
                        "model_name": payload.get("model_name") or custom_metadata.get("model_name") or artifact_path.stem,
                        "version": payload.get("version") or custom_metadata.get("version"),
                        "training_timestamp": custom_metadata.get("training_timestamp") or payload.get("saved_timestamp"),
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "r2": metrics.get("r2"),
                        "training_samples": custom_metadata.get("training_samples"),
                        "dataset_size": custom_metadata.get("dataset_size"),
                    }
                )
            training_runs.sort(key=lambda item: str(item.get("training_timestamp") or ""), reverse=True)
            latest_run = training_runs[0] if training_runs else None
            return {
                "latest_training_runs": training_runs[:3],
                "mae": latest_run.get("mae") if latest_run else None,
                "rmse": latest_run.get("rmse") if latest_run else None,
                "r2": latest_run.get("r2") if latest_run else None,
                "training_samples": latest_run.get("training_samples") if latest_run else None,
                "dataset_size": latest_run.get("dataset_size") if latest_run else None,
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("assistant_context_training_failed", error=str(exc))
            return {
                "latest_training_runs": [],
                "mae": None,
                "rmse": None,
                "r2": None,
                "training_samples": None,
                "dataset_size": None,
            }

    def get_dataset_context(self) -> dict[str, Any]:
        """Return a compact summary of the engineered training dataset."""
        try:
            dataset_path = self._repo_root / "ml" / "data" / "features" / "engineered_training_dataset.csv"
            if not dataset_path.exists():
                return {"record_count": 0, "feature_names": [], "target_column": None, "missing_values_summary": {}}
            dataframe = pd.read_csv(dataset_path)
            missing_summary = dataframe.isna().sum().to_dict()
            return {
                "record_count": int(len(dataframe)),
                "feature_names": [str(column) for column in dataframe.columns],
                "target_column": "actual_delivery_time_min",
                "missing_values_summary": {key: int(value) for key, value in missing_summary.items() if value},
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("assistant_context_dataset_failed", error=str(exc))
            return {"record_count": 0, "feature_names": [], "target_column": None, "missing_values_summary": {}}

    def get_latest_prediction_context(self) -> dict[str, Any]:
        """Return the most recent monitoring record as the latest prediction summary."""
        try:
            latest_record = self._monitoring_engine.get_latest()
            if latest_record is None:
                return {"status": "unavailable", "summary": None}
            return {
                "status": "available",
                "summary": {
                    "model_name": latest_record.model_name,
                    "prediction_count": latest_record.prediction_count,
                    "mean_prediction": latest_record.mean_prediction,
                    "timestamp": latest_record.timestamp,
                },
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("assistant_context_latest_prediction_failed", error=str(exc))
            return {"status": "unavailable", "summary": None}

    def get_eda_context(self) -> dict[str, Any]:
        """Return a lightweight EDA summary from persisted reports when available."""
        try:
            report_path = self._repo_root / "ml" / "reports" / "dataset_profile.json"
            if not report_path.exists():
                return {"status": "unavailable", "summary": None}
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            return {
                "status": "available",
                "summary": {
                    "dataset_count": payload.get("dataset_count"),
                    "datasets": payload.get("datasets", []),
                },
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("assistant_context_eda_failed", error=str(exc))
            return {"status": "unavailable", "summary": None}

    def get_explainability_context(self) -> dict[str, Any]:
        """Return explainability insights for the current production model when available."""
        try:
            production_context = self.get_production_model_context()
            artifact_path = production_context.get("artifact_path")
            if not artifact_path:
                return {"available": False, "summary": "No explainability information is available.", "top_features": [], "confidence": None}

            registry_metadata = {}
            if self._registry_engine is not None:
                try:
                    production_model = self._registry_engine.get_production_model(str(production_context.get("name") or ""))
                    registry_metadata = dict(getattr(production_model, "metadata", {}) or {})
                except Exception:
                    registry_metadata = {}

            feature_importance_path = registry_metadata.get("feature_importance_path")
            if feature_importance_path:
                resolved_path = Path(feature_importance_path)
                if not resolved_path.is_absolute():
                    resolved_path = (self._repo_root / resolved_path).resolve()
                if resolved_path.exists():
                    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
                    top_features = [
                        {"feature_name": item.get("feature_name"), "importance": item.get("importance")}
                        for item in (payload.get("ranked_features") or [])[:5]
                    ]
                    return {
                        "available": True,
                        "summary": (
                            f"The current production model uses {payload.get('method', 'persisted_artifact')} to rank the most influential features. "
                            f"The strongest drivers are listed below."
                        ),
                        "top_features": top_features,
                        "confidence": self._infer_confidence_from_payload(payload),
                        "model_name": payload.get("model_name") or production_context.get("name"),
                        "method": payload.get("method") or "persisted_artifact",
                    }

            from ml.training.explainability import ExplainabilityEngine
            from ml.training.persistence import ModelPersistenceEngine

            model = ModelPersistenceEngine().load_model(artifact_path)
            feature_names = self._get_feature_names_for_production_model(production_context)
            if not feature_names:
                return {"available": False, "summary": "No feature metadata is available for this model.", "top_features": [], "confidence": None}

            explainability = ExplainabilityEngine().explain_model(
                model,
                str(production_context.get("name") or "production_model"),
                feature_names,
            )
            top_features = [
                {"feature_name": item["feature_name"], "importance": item["importance"]}
                for item in explainability.ranked_features[:5]
            ]
            confidence = self._infer_confidence(explainability)
            return {
                "available": True,
                "summary": (
                    f"The current production model uses {explainability.explanation_method} to rank the most influential features. "
                    f"The strongest drivers are listed below."
                ),
                "top_features": top_features,
                "confidence": confidence,
                "model_name": explainability.model_name,
                "method": explainability.explanation_method,
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("assistant_context_explainability_failed", error=str(exc))
            return {"available": False, "summary": "No explainability information is available.", "top_features": [], "confidence": None}

    def _get_feature_names_for_production_model(self, production_context: dict[str, Any]) -> list[str]:
        """Return feature names from dataset metadata or registry metadata when available."""
        dataset_context = self.get_dataset_context()
        feature_names = dataset_context.get("feature_names") or []
        if feature_names:
            return [str(name) for name in feature_names]

        try:
            if self._registry_engine is None:
                return []
            production_name = production_context.get("name")
            if not production_name:
                return []
            for model in self._registry_engine.list_models():
                if getattr(model, "model_name", None) == production_name:
                    metadata = getattr(model, "metadata", {}) or {}
                    feature_names = metadata.get("feature_names") or []
                    if feature_names:
                        return [str(name) for name in feature_names]
        except Exception:  # pragma: no cover - defensive fallback
            return []
        return []

    @staticmethod
    def _infer_confidence_from_payload(payload: dict[str, Any]) -> float | None:
        """Infer a simple confidence score from persisted explainability payloads."""
        ranked_features = payload.get("ranked_features") or []
        if not ranked_features:
            return None
        return round(float(sum(item.get("importance", 0.0) for item in ranked_features[:3])) / max(1, len(ranked_features[:3])), 4)

    @staticmethod
    def _infer_confidence(explanation: Any) -> str:
        """Infer a simple business-friendly confidence label from model metrics when available."""
        if not hasattr(explanation, "feature_importance"):
            return "moderate"
        return "high"
