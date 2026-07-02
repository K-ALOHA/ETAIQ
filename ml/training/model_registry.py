"""Production model registry for ETAIQ model lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_config import TrainingLogger


@dataclass
class RegisteredModel:
    """A persisted model artifact registered for production workflows."""

    model_name: str
    version: int
    artifact_path: Path
    metrics: dict[str, Any]
    created_at: str
    status: str


class ModelRegistryEngine:
    """Manage model registration, lifecycle, and production promotion."""

    def __init__(self, logger: TrainingLogger | None = None, storage_dir: str | Path | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.model_registry")
        self._storage_dir = Path(storage_dir or Path("ml/data/training/model_registry"))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[tuple[str, int], RegisteredModel] = {}

    def register_model(
        self,
        model_name: str,
        version: int,
        artifact_path: str | Path,
        metrics: dict[str, Any],
        status: str,
    ) -> RegisteredModel:
        """Register a model artifact and store its metadata in memory."""
        if not model_name:
            raise ValueError("model_name cannot be empty")
        if version < 1:
            raise ValueError("version must be at least 1")

        resolved_path = Path(artifact_path)
        if not resolved_path.exists():
            raise ValueError(f"missing artifact: {resolved_path}")

        key = (model_name, version)
        if key in self._models:
            raise ValueError(f"duplicate model registration: {model_name} v{version}")

        record = RegisteredModel(
            model_name=model_name,
            version=version,
            artifact_path=resolved_path,
            metrics=dict(metrics),
            created_at=datetime.now(timezone.utc).isoformat(),
            status=status,
        )
        self._models[key] = record
        self._logger.info("Model registered", model_name=model_name, version=version, status=status)
        return RegisteredModel(
            model_name=record.model_name,
            version=record.version,
            artifact_path=record.artifact_path,
            metrics=dict(record.metrics),
            created_at=record.created_at,
            status=record.status,
        )

    def list_models(self) -> list[RegisteredModel]:
        """Return all registered models."""
        return list(self._models.values())

    def get_model(self, model_name: str, version: int) -> RegisteredModel:
        """Return a registered model by name and version."""
        if not model_name:
            raise ValueError("model_name cannot be empty")
        key = (model_name, version)
        if key not in self._models:
            raise ValueError(f"unknown model: {model_name} v{version}")
        return self._models[key]

    def set_production(self, model_name: str, version: int) -> RegisteredModel:
        """Promote a model to Production and archive any previous Production model."""
        if not model_name:
            raise ValueError("model_name cannot be empty")

        target = self.get_model(model_name, version)
        for other in self._models.values():
            if other.model_name == model_name and other.status == "Production" and other.version != version:
                other.status = "Archived"
                self._logger.info("Model archived", model_name=other.model_name, version=other.version)

        target.status = "Production"
        self._logger.info("Production switched", model_name=model_name, version=version)
        return target

    def archive_model(self, model_name: str, version: int) -> RegisteredModel:
        """Archive a registered model."""
        record = self.get_model(model_name, version)
        record.status = "Archived"
        self._logger.info("Model archived", model_name=model_name, version=version)
        return RegisteredModel(
            model_name=record.model_name,
            version=record.version,
            artifact_path=record.artifact_path,
            metrics=dict(record.metrics),
            created_at=record.created_at,
            status=record.status,
        )
