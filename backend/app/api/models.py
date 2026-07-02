"""Model registry management API routes for ETAIQ."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.logging import get_logger
from app.schemas.ml_management import ModelRegistryResponse, RegisteredModelResponse
from ml.training.model_registry import ModelRegistryEngine

logger = get_logger(__name__)
router = APIRouter(tags=["models"])
registry_engine = ModelRegistryEngine()


@router.get(
    "/models/registry",
    response_model=ModelRegistryResponse,
    summary="List registered models",
    description="Returns the current model registry entries.",
)
async def list_models() -> ModelRegistryResponse:
    """Return the current registered models from the shared registry engine."""
    logger.info("models_requested", endpoint="/api/v1/models/registry")

    try:
        models = registry_engine.list_models()
        return ModelRegistryResponse(
            models=[
                RegisteredModelResponse(
                    model_name=model.model_name,
                    version=model.version,
                    artifact_path=str(model.artifact_path),
                    metrics=model.metrics,
                    created_at=model.created_at,
                    status=model.status,
                )
                for model in models
            ],
            count=len(models),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("models_failed", endpoint="/api/v1/models", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="unable to list models") from exc
