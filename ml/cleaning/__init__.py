"""ETAIQ Cleaning Execution Engine package."""

from __future__ import annotations

__all__ = ["CleaningEngine", "run_cleaning", "execute_rollback"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from ml.cleaning.cleaning_engine import CleaningEngine
        from ml.cleaning.main import execute_rollback, run_cleaning

        return {
            "CleaningEngine": CleaningEngine,
            "run_cleaning": run_cleaning,
            "execute_rollback": execute_rollback,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
