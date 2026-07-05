"""Central API router aggregating versioned endpoint modules."""

from importlib import import_module

from fastapi import APIRouter

from app.core.config import get_settings

settings = get_settings()


def _build_api_router() -> APIRouter:
    """Create the aggregated API router without importing optional training dependencies eagerly."""
    router = APIRouter(prefix=settings.api_prefix)

    router.include_router(import_module("app.api.assistant").router)
    router.include_router(import_module("app.api.prediction").router)
    router.include_router(import_module("app.api.training").router)
    router.include_router(import_module("app.api.models").router)
    router.include_router(import_module("app.api.experiments").router)
    router.include_router(import_module("app.api.monitoring").router)
    router.include_router(import_module("app.api.drift").router)
    router.include_router(import_module("app.api.explainability").router)
    router.include_router(import_module("app.api.dataset").router)
    router.include_router(import_module("app.api.performance").router)
    return router


api_router = _build_api_router()
