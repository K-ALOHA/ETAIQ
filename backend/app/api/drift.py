"""Drift detection management API routes for ETAIQ."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.logging import get_logger
from app.schemas.ml_management import DriftResponse, DriftResultResponse
from ml.training.drift_detection import DriftDetectionEngine

logger = get_logger(__name__)
router = APIRouter(tags=["drift"])
drift_engine = DriftDetectionEngine()


@router.get(
    "/drift",
    response_model=DriftResponse,
    summary="Get latest drift detection results",
    description="Returns the latest drift results from the existing drift detector.",
)
async def get_drift() -> DriftResponse:
    """Return the latest drift result payload from the existing drift engine."""
    logger.info("drift_requested", endpoint="/api/v1/drift")

    try:
        if drift_engine._baseline is None:
            return DriftResponse(results=[], drift_detected=False)

        results = drift_engine.detect_drift(drift_engine._baseline)
        return DriftResponse(
            results=[
                DriftResultResponse(
                    feature_name=result.feature_name,
                    baseline_mean=result.baseline_mean,
                    current_mean=result.current_mean,
                    baseline_std=result.baseline_std,
                    current_std=result.current_std,
                    drift_score=result.drift_score,
                    drift_detected=result.drift_detected,
                )
                for result in results
            ],
            drift_detected=any(result.drift_detected for result in results),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("drift_failed", endpoint="/api/v1/drift", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="unable to detect drift"
        ) from exc
