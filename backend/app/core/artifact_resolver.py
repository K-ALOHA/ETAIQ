"""Resolve model artifact paths independent of where they were originally stored.

Resolution rules:
- Absolute stored path  → returned as-is (caller validates existence).
- Relative stored path  → joined with the configured MODEL_ARTIFACT_DIR.

This makes test fixtures self-contained (they register absolute tmp_path
paths that are returned unchanged) while keeping Docker / production
behaviour unchanged (the registry stores only filenames, which are relative
and therefore resolved through the configured artifact directory).
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

    If *stored_path* is absolute, return it directly — the caller is
    responsible for checking existence.  This preserves test isolation
    when fixtures register models under tmp_path.

    If *stored_path* is relative (as stored by the production registry,
    which persists only the filename), join it with the configured
    MODEL_ARTIFACT_DIR so the result is valid in every environment
    (host macOS, Docker /app/ml/, CI clean checkout).
    """
    path = Path(stored_path)
    if path.is_absolute():
        return path
    return _artifact_dir() / path.name
