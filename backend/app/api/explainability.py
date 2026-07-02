"""Explainability management API routes for ETAIQ."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.ml_management import ExplainabilityResponse
from ml.training.explainability import ExplainabilityEngine
from ml.training.persistence import ModelPersistenceEngine

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(tags=["explainability"])


@router.get(
    "/explainability/{model_name}",
    response_model=ExplainabilityResponse,
    summary="Get explainability output for a model",
    description="Returns the explainability output derived from the existing explainability engine.",
)
async def get_explainability(model_name: str) -> ExplainabilityResponse:
    """Return explainability output for the requested model using the existing explainability engine."""
    logger.info("explainability_requested", endpoint="/api/v1/explainability", model_name=model_name)

    try:
        models_dir = Path(settings.model_path)
        model_path = next((path for path in sorted(models_dir.glob(f"{model_name}_v*.joblib")) if path.exists()), None)
        if model_path is None:
            raise FileNotFoundError(f"Model not found: {model_name}")

        model = ModelPersistenceEngine().load_model(model_path)
        engine = ExplainabilityEngine()
        result = engine.explain_model(model, model_name, ["feature_0"])
        return ExplainabilityResponse(
            model_name=result.model_name,
            feature_importance=result.feature_importance,
            ranked_features=result.ranked_features,
            explanation_time_seconds=result.explanation_time_seconds,
            explanation_method=result.explanation_method,
            sample_count=result.sample_count,
        )
    except FileNotFoundError as exc:
        logger.error("model_not_found", endpoint="/api/v1/explainability", error=str(exc))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("explainability_failed", endpoint="/api/v1/explainability", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="unable to generate explainability") from exc
