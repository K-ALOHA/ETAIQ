"""Experiment tracking management API routes for ETAIQ."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.logging import get_logger
from app.schemas.ml_management import ExperimentHistoryResponse, ExperimentResponse
from ml.training.experiment_tracking import ExperimentTrackingEngine

logger = get_logger(__name__)
router = APIRouter(tags=["experiments"])
experiment_engine = ExperimentTrackingEngine()


@router.get(
    "/experiments",
    response_model=ExperimentHistoryResponse,
    summary="List experiment history",
    description="Returns the current experiment tracking history.",
)
async def list_experiments() -> ExperimentHistoryResponse:
    """Return recorded experiments from the existing experiment tracking engine."""
    logger.info("experiments_requested", endpoint="/api/v1/experiments")

    try:
        experiments = experiment_engine.list_experiments()
        return ExperimentHistoryResponse(
            experiments=[
                ExperimentResponse(
                    experiment_id=record.experiment_id,
                    timestamp=record.timestamp,
                    model_name=record.model_name,
                    dataset_version=record.dataset_version,
                    hyperparameters=record.hyperparameters,
                    metrics=record.metrics,
                    training_time_seconds=record.training_time_seconds,
                    model_version=record.model_version,
                )
                for record in experiments
            ],
            count=len(experiments),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("experiments_failed", endpoint="/api/v1/experiments", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="unable to list experiments") from exc
