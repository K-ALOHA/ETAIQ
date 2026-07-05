"""API route modules."""

from importlib import import_module

__all__ = ["api_router"]


def __getattr__(name: str):
    """Lazily expose the aggregated API router without importing all route modules eagerly."""
    if name == "api_router":
        return import_module("app.api.router").api_router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
