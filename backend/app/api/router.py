"""Central API router aggregating versioned endpoint modules."""

from fastapi import APIRouter

from app.api.drift import router as drift_router
from app.api.experiments import router as experiments_router
from app.api.explainability import router as explainability_router
from app.api.models import router as models_router
from app.api.monitoring import router as monitoring_router
from app.api.prediction import router as prediction_router
from app.api.training import router as training_router
from app.core.config import get_settings

settings = get_settings()

api_router = APIRouter(prefix=settings.api_prefix)
api_router.include_router(prediction_router)
api_router.include_router(training_router)
api_router.include_router(models_router)
api_router.include_router(experiments_router)
api_router.include_router(monitoring_router)
api_router.include_router(drift_router)
api_router.include_router(explainability_router)
