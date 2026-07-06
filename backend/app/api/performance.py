"""Performance metrics API routes for ETAIQ."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.models import registry_engine
from app.core.logging import get_logger
from app.schemas.ml_management import PerformanceResponse

logger = get_logger(__name__)
router = APIRouter(tags=["performance"])


@router.get(
    "/performance",
    response_model=PerformanceResponse,
    summary="Get performance metrics",
    description="Returns performance metrics of the production model.",
)
async def get_performance() -> PerformanceResponse:
    """Return performance metrics."""
    logger.info("performance_requested", endpoint="/api/v1/performance")

    try:
        try:
            production_model = registry_engine.select_production_model("XGBRegressor")
            metrics = production_model.metrics
            mae = metrics.get("mae")
            rmse = metrics.get("rmse")
            mape = metrics.get("mape")
            r2 = metrics.get("r2")
        except ValueError:
            mae = None
            rmse = None
            mape = None
            r2 = None
        
        # Get inference latency (we can use monitoring records if available)
        from ml.training.monitoring import MonitoringEngine
        monitoring = MonitoringEngine()
        records = monitoring.list_records()
        inference_latency_ms = None
        for record in records:
            for key in ("latency_ms", "prediction_latency_ms", "latency"):
                val = getattr(record, key, None)
                if isinstance(val, int | float):
                    inference_latency_ms = float(val)
                    break
            if inference_latency_ms is not None:
                break

        return PerformanceResponse(
            mae=mae,
            rmse=rmse,
            mape=mape,
            r2=r2,
            inference_latency_ms=inference_latency_ms,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("performance_failed", endpoint="/api/v1/performance", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="unable to get performance metrics",
        ) from exc
