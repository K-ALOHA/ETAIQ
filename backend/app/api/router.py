"""Central API router aggregating versioned endpoint modules."""

from fastapi import APIRouter

from app.api.prediction import router as prediction_router
from app.core.config import get_settings

settings = get_settings()

api_router = APIRouter(prefix=settings.api_prefix)
api_router.include_router(prediction_router)
