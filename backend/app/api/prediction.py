"""Prediction API routes for serving persisted ETAIQ models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status

from app.api.models import registry_engine
from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.prediction import HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse
from ml.training.prediction_pipeline import PredictionPipelineEngine

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(tags=["prediction"])


def _resolve_model_path(model_name: str | None = None) -> Path:
    """Resolve the model artifact path from the configured directory."""
    models_dir = Path(settings.model_path)
    models_dir.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        candidates = sorted(models_dir.glob("*.joblib"))
        if not candidates:
            raise FileNotFoundError("No model artifacts found")
        return candidates[-1]

    candidate = models_dir / f"{model_name}.joblib"
    if candidate.exists():
        return candidate

    versioned_candidate = sorted(models_dir.glob(f"{model_name}_v*.joblib"))
    if versioned_candidate:
        return versioned_candidate[-1]

    raise FileNotFoundError(f"Model not found: {model_name}")


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
        model_path = _resolve_model_path()
        pipeline = PredictionPipelineEngine(logger=None)
        features_frame = pd.DataFrame([validated_payload])
        result = pipeline.predict(model_path, features_frame)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            "prediction_completed",
            endpoint="/api/v1/predict",
            model_name=result.model_name,
            model_version=result.model_version,
            latency_ms=round(elapsed_ms, 3),
        )

        return PredictionResponse(
            prediction=float(result.predictions[0]),
            model_name=result.model_name,
            model_version=result.model_version,
            processing_time_ms=round(elapsed_ms, 3),
        )
    except FileNotFoundError as exc:
        logger.error("model_not_found", endpoint="/api/v1/predict", error=str(exc))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
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

    models_dir = Path(settings.model_path)
    model_files = sorted(models_dir.glob("*.joblib"))
    if not model_files:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="model not found")

    latest_model = model_files[-1]
    metadata_path = latest_model.with_suffix(".json")
    created_at = latest_model.stat().st_mtime
    if metadata_path.exists():
        created_at = metadata_path.stat().st_mtime

    registered_models = registry_engine.list_models()

    return ModelInfoResponse(
        current_model=latest_model.stem,
        version=int(latest_model.stem.split("_v", maxsplit=1)[1]) if "_v" in latest_model.stem else 1,
        created_at=str(created_at),
        available_models=[path.stem for path in model_files],
        models=[
            {
                "model_name": model.model_name,
                "version": model.version,
                "artifact_path": str(model.artifact_path),
                "status": model.status,
            }
            for model in registered_models
        ],
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
    models_dir = Path(settings.model_path)
    models_dir.mkdir(parents=True, exist_ok=True)
    has_model = any(models_dir.glob("*.joblib"))
    return HealthResponse(status="healthy", model_loaded=has_model)
