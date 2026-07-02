"""Monitoring management API routes for ETAIQ."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.logging import get_logger
from app.schemas.ml_management import MonitoringRecordResponse, MonitoringResponse
from ml.training.monitoring import MonitoringEngine

logger = get_logger(__name__)
router = APIRouter(tags=["monitoring"])
monitoring_engine = MonitoringEngine()


@router.get(
    "/monitoring",
    response_model=MonitoringResponse,
    summary="List monitoring records",
    description="Returns monitoring records captured by the existing monitoring engine.",
)
async def list_monitoring() -> MonitoringResponse:
    """Return monitoring records from the existing monitoring engine."""
    logger.info("monitoring_requested", endpoint="/api/v1/monitoring")

    try:
        records = monitoring_engine.list_records()
        return MonitoringResponse(
            records=[
                MonitoringRecordResponse(
                    timestamp=record.timestamp,
                    model_name=record.model_name,
                    prediction_count=record.prediction_count,
                    mean_prediction=record.mean_prediction,
                    std_prediction=record.std_prediction,
                    min_prediction=record.min_prediction,
                    max_prediction=record.max_prediction,
                    missing_inputs=record.missing_inputs,
                    out_of_range_inputs=record.out_of_range_inputs,
                )
                for record in records
            ],
            count=len(records),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("monitoring_failed", endpoint="/api/v1/monitoring", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="unable to list monitoring records") from exc
