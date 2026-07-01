"""Central API router aggregating versioned endpoint modules."""

from fastapi import APIRouter

from app.core.config import get_settings

settings = get_settings()

api_router = APIRouter(prefix=settings.api_prefix)
