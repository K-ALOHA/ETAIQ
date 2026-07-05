"""Prediction API routes for serving persisted ETAIQ models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Request, status

from app.api.models import registry_engine
from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.prediction import HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse
from ml.training.prediction_pipeline import PredictionPipelineEngine
from ml.training.monitoring import MonitoringEngine

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(tags=["prediction"])


def _resolve_latest_artifact() -> Path:
    """Return the latest model artifact found in the configured model directory."""
    models_dir = Path(settings.model_path)
    models_dir.mkdir(parents=True, exist_ok=True)
    # search recursively to support nested `models/` folder layout
    candidates = sorted(models_dir.rglob("*.joblib"))
    if not candidates:
        raise FileNotFoundError("no model artifacts found")
    return candidates[-1]


def _resolve_production_model() -> Path:
    """Resolve the latest Production XGBRegressor artifact from the model registry."""
    selected = registry_engine.select_production_model("XGBRegressor")
    model_path = selected.artifact_path

    if not model_path.exists():
        raise FileNotFoundError(f"Production artifact is missing: {model_path}")

    return model_path


def _read_artifact_metadata(model_path: Path) -> dict[str, Any]:
    """Read persisted model metadata from the sibling JSON file when present."""
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if isinstance(payload.get("custom_metadata"), dict):
        return dict(payload["custom_metadata"])
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _validate_feature_payload(payload: dict[str, object] | None) -> dict[str, Any]:
    """Validate the feature payload and return a normalized dictionary."""
    if payload is None:
        raise ValueError("missing features")
    if not isinstance(payload, dict):
        raise ValueError("features must be an object")
    if not payload:
        raise ValueError("features cannot be empty")

    for feature_name, feature_value in payload.items():
        if not isinstance(feature_name, str):
            raise ValueError("feature names must be strings")
        if feature_value is None:
            raise ValueError("feature values cannot be null")
        if isinstance(feature_value, (dict, list, tuple, set)):
            raise ValueError("feature values must be scalar values")

    return {str(key): value for key, value in payload.items()}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Run a prediction for a single feature set",
    description="Loads the latest persisted model and returns a prediction.",
)
async def predict(request: Request) -> PredictionResponse:
    """Serve a prediction for a single input feature payload."""
    logger.info("request_received", endpoint="/api/v1/predict")

    try:
        payload = await request.json()
    except ValueError as exc:
        logger.error("invalid_json", endpoint="/api/v1/predict", error=str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid payload") from exc

    try:
        validated_payload = _validate_feature_payload(payload.get("features") if isinstance(payload, dict) else None)
    except ValueError as exc:
        logger.error("invalid_request", endpoint="/api/v1/predict", error=str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        start_time = time.perf_counter()
        production_model = registry_engine.select_production_model("XGBRegressor")
        model_path = production_model.artifact_path
        if not model_path.exists():
            raise FileNotFoundError(f"Production artifact is missing: {model_path}")

        pipeline = PredictionPipelineEngine(logger=None)
        features_frame = pd.DataFrame([validated_payload])
        try:
            result = pipeline.predict(model_path, features_frame)
        except Exception as exc:
            logger.warning(
                "pipeline_prediction_failed",
                endpoint="/api/v1/predict",
                error=str(exc),
                model_path=str(model_path),
            )
            try:
                trained_model = pipeline._persistence_engine.load_model(model_path)
                if not hasattr(trained_model, "predict"):
                    raise TypeError("loaded model has no predict method")

                values = [float(value) for value in validated_payload.values()]
                if not values:
                    values = [0.0]

                preds = trained_model.predict(np.asarray([values], dtype=float))

                class _R:
                    pass

                result = _R()
                result.predictions = np.asarray(preds).reshape(-1)
                result.model_name = pipeline._inference_engine._extract_model_name(model_path)
                result.model_version = pipeline._inference_engine._extract_model_version(model_path)
            except Exception as fallback_error:
                logger.warning(
                    "direct_prediction_fallback_failed",
                    endpoint="/api/v1/predict",
                    error=str(fallback_error),
                    model_path=str(model_path),
                )
                raise exc from fallback_error
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Record this prediction for monitoring and explainability
        monitoring_engine = MonitoringEngine(load_existing_records=True)
        monitoring_engine.record_predictions(
            model_name=result.model_name,
            predictions=result.predictions,
        )

        logger.info(
            "prediction_completed",
            endpoint="/api/v1/predict",
            model_name=result.model_name,
            model_version=result.model_version,
            registry_status=production_model.status,
            artifact_path=str(model_path),
            load_time_ms=round(elapsed_ms, 3),
            latency_ms=round(elapsed_ms, 3),
        )

        return PredictionResponse(
            prediction=float(result.predictions[0]),
            model_name=result.model_name,
            model_version=result.model_version,
            processing_time_ms=round(elapsed_ms, 3),
        )
    except LookupError as exc:
        logger.error("registry_no_production_model", endpoint="/api/v1/predict", error=str(exc))
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        logger.error("registry_no_production_xgb_model", endpoint="/api/v1/predict", error=str(exc))
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error("production_artifact_missing", endpoint="/api/v1/predict", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("registry_multiple_production_models", endpoint="/api/v1/predict", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except ValueError as exc:
        logger.error("invalid_request", endpoint="/api/v1/predict", error=str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("prediction_failed", endpoint="/api/v1/predict", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="prediction failed") from exc


@router.get(
    "/models",
    response_model=ModelInfoResponse,
    summary="List available persisted models",
    description="Returns the legacy model metadata payload expected by the prediction API and includes registry details.",
)
async def list_models() -> ModelInfoResponse:
    """Return the legacy model metadata response while exposing registry details for management endpoints."""
    logger.info("models_requested", endpoint="/api/v1/models")

    # Prefer the latest Production XGBRegressor model for dashboard and model metadata.
    try:
        selected = registry_engine.select_production_model("XGBRegressor")
        current_model = selected.artifact_path.stem
        version = selected.version
        created_at = selected.created_at
    except ValueError:
        try:
            artifact = _resolve_latest_artifact()
            current_model = artifact.stem
            version = int(artifact.stem.split("_v", maxsplit=1)[1]) if "_v" in artifact.stem else 1
            created_at = str(artifact.stat().st_mtime)
        except FileNotFoundError:
            current_model = ""
            version = 0
            created_at = ""

    registered_models = registry_engine.list_models()
    model_payloads = []
    for model in registered_models:
        metadata = dict(model.metadata or {})
        artifact_metadata = _read_artifact_metadata(model.artifact_path)
        metadata = {**artifact_metadata, **metadata}

        model_payloads.append(
            {
                "model_name": model.model_name,
                "version": model.version,
                "artifact_path": str(model.artifact_path),
                "status": model.status,
                "metrics": model.metrics,  # Include metrics for dashboard display
                "created_at": model.created_at,
                "dataset_size": metadata.get("dataset_size"),
                "training_samples": metadata.get("training_samples"),
                "testing_samples": metadata.get("testing_samples"),
                "feature_names": metadata.get("feature_names") or [],
                "feature_count": metadata.get("feature_count"),
                "target_column": metadata.get("target_column"),
            }
        )

    return ModelInfoResponse(
        current_model=current_model,
        version=version,
        created_at=created_at,
        available_models=[str(model.artifact_path.stem) for model in registered_models] if registered_models else [artifact.stem] if (current_model and current_model) else [],
        models=model_payloads,
        count=len(registered_models),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Prediction service health",
    description="Returns the health status and whether a model is loaded.",
)
async def health() -> HealthResponse:
    """Report whether the prediction service has a model available."""
    logger.info("health_requested", endpoint="/api/v1/health")
    try:
        _resolve_production_model()
        has_model = True
    except FileNotFoundError:
        has_model = False
    return HealthResponse(status="healthy", model_loaded=has_model)
