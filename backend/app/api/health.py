"""Health check API routes."""

from fastapi import APIRouter

from app.core.logging import get_logger
from app.schemas.health import HealthResponse

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns the current operational status of the ETAIQ API.",
)
async def health_check() -> HealthResponse:
    """Verify that the API service is running and responsive.

    Returns:
        HealthResponse: Object containing the service health status.
    """
    logger.info("health_check_requested")
    return HealthResponse(status="healthy")
