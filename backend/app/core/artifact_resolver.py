"""Resolve model artifact paths independent of where they were originally stored.

The registry may contain absolute host paths (e.g. /Users/kaloha/ETAIQ/ml/...)
that are invalid inside Docker (/app/ml/...).  This module provides a single
resolver that always rebuilds the path from the configured artifact directory
using only the filename, making loading environment-agnostic.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings


def _artifact_dir() -> Path:
    """Return the resolved absolute path to the model artifact directory.

    Resolution order:
    1. If MODEL_ARTIFACT_DIR is already absolute, use it directly.
    2. If relative, try resolving from cwd (works in Docker where cwd=/app).
    3. If that path doesn't exist, walk up from __file__ to find the repo root
       (works in local dev where the backend is run from backend/).
    """
    settings = get_settings()
    configured = Path(settings.model_artifact_dir)

    if configured.is_absolute():
        return configured

    # Try cwd-relative first (Docker: cwd=/app -> /app/ml/artifacts/models)
    # noqa: E501 — inline comment, cannot be shortened without losing meaning
    cwd_candidate = (Path.cwd() / configured).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    # Fallback: walk up from this file to find a directory that contains ml/artifacts/models
    # __file__ = .../app/core/artifact_resolver.py
    for parent in Path(__file__).resolve().parents:
        candidate = (parent / configured).resolve()
        if candidate.exists():
            return candidate

    # Last resort: return the cwd-relative path even if it doesn't exist yet
    return cwd_candidate


def resolve_artifact_path(stored_path: str | Path) -> Path:
    """Return the correct on-disk path for a model artifact.

    Takes only the filename from *stored_path* and joins it with the
    configured MODEL_ARTIFACT_DIR, so the result is always valid in the
    current environment regardless of where the path was originally recorded.
    """
    filename = Path(stored_path).name
    return _artifact_dir() / filename
