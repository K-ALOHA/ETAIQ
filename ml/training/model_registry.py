"""Production model registry for ETAIQ model lifecycle management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistryEngine:
    """Manage model registration, lifecycle, and production promotion."""

    def __init__(self, logger: TrainingLogger | None = None, storage_dir: str | Path | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.model_registry")
        resolved_storage_dir = Path(storage_dir or "ml/data/training/model_registry")
        if not resolved_storage_dir.is_absolute():
            resolved_storage_dir = (Path(__file__).resolve().parents[2] / resolved_storage_dir).resolve()
        self._storage_dir = resolved_storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[tuple[str, int], RegisteredModel] = {}
        self._load_existing_registry()

    def register_model(
        self,
        model_name: str,
        version: int,
        artifact_path: str | Path,
        metrics: dict[str, Any],
        status: str,
        metadata: dict[str, Any] | None = None,
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
            metadata=dict(metadata or {}),
        )
        self._models[key] = record
        self._persist_registration(record)
        self._logger.info("Model registered", model_name=model_name, version=version, status=status)
        return RegisteredModel(
            model_name=record.model_name,
            version=record.version,
            artifact_path=record.artifact_path,
            metrics=dict(record.metrics),
            created_at=record.created_at,
            status=record.status,
            metadata=dict(record.metadata),
        )

    def list_models(self) -> list[RegisteredModel]:
        """Return all registered models."""
        return list(self._models.values())

    def get_production_model(self, model_name: str) -> RegisteredModel:
        """Return the latest active Production model for the given model name."""
        if not model_name:
            raise ValueError("model_name cannot be empty")

        production_models = [
            record for record in self._models.values()
            if record.model_name == model_name and record.status == "Production"
        ]
        if not production_models:
            raise ValueError(f"no production model registered for: {model_name}")
        return max(production_models, key=lambda record: (record.version, record.created_at))

    def select_production_model(self, preferred_model_name: str | None = None) -> RegisteredModel:
        """Select the latest Production model, optionally requiring a specific model family."""
        production_models = [record for record in self.list_models() if record.status == "Production"]
        if not production_models:
            raise ValueError("No production model is registered.")

        if preferred_model_name:
            requested_name = str(preferred_model_name).strip()
            requested_models = [record for record in production_models if str(record.model_name).lower() == requested_name.lower()]
            if not requested_models:
                raise ValueError(f"No Production {requested_name} model is registered.")
            production_models = requested_models

        return max(production_models, key=lambda record: (record.version, record.created_at))

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
                self._persist_registration(other)
                self._logger.info("Model archived", model_name=other.model_name, version=other.version)

        target.status = "Production"
        self._persist_registration(target)
        self._logger.info("Production switched", model_name=model_name, version=version)
        return target

    def archive_model(self, model_name: str, version: int) -> RegisteredModel:
        """Archive a registered model."""
        record = self.get_model(model_name, version)
        record.status = "Archived"
        self._persist_registration(record)
        self._logger.info("Model archived", model_name=model_name, version=version)
        return RegisteredModel(
            model_name=record.model_name,
            version=record.version,
            artifact_path=record.artifact_path,
            metrics=dict(record.metrics),
            created_at=record.created_at,
            status=record.status,
            metadata=dict(record.metadata),
        )

    def update_explainability_metadata(
        self,
        *,
        model_name: str,
        version: int,
        explainability_dir: str | Path,
        feature_importance_path: str | Path | None = None,
        local_explanation_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        shap_path: str | Path | None = None,
        summary_plot_path: str | Path | None = None,
        waterfall_plot_path: str | Path | None = None,
    ) -> RegisteredModel:
        """Attach explainability artifact locations to an existing registered model."""
        record = self.get_model(model_name, version)
        metadata = dict(record.metadata or {})
        metadata.update(
            {
                "explainability_available": True,
                "explainability_dir": str(Path(explainability_dir)),
                "feature_importance_path": str(feature_importance_path) if feature_importance_path else metadata.get("feature_importance_path"),
                "local_explanation_path": str(local_explanation_path) if local_explanation_path else metadata.get("local_explanation_path"),
                "metadata_path": str(metadata_path) if metadata_path else metadata.get("metadata_path"),
                "shap_path": str(shap_path) if shap_path else metadata.get("shap_path"),
                "summary_plot_path": str(summary_plot_path) if summary_plot_path else metadata.get("summary_plot_path"),
                "waterfall_plot_path": str(waterfall_plot_path) if waterfall_plot_path else metadata.get("waterfall_plot_path"),
            }
        )
        record.metadata = metadata
        self._persist_registration(record)
        self._logger.info("Explainability metadata updated", model_name=model_name, version=version)
        return RegisteredModel(
            model_name=record.model_name,
            version=record.version,
            artifact_path=record.artifact_path,
            metrics=dict(record.metrics),
            created_at=record.created_at,
            status=record.status,
            metadata=dict(record.metadata),
        )

    def _persist_registration(self, record: RegisteredModel) -> None:
        """Persist registration metadata to disk."""
        output_path = self._storage_dir / f"{record.model_name}_v{record.version}_registry.json"
        
        # Store relative path for portability across machines/environments
        try:
            relative_path = record.artifact_path.relative_to(self._storage_dir.parent.parent.parent)
        except ValueError:
            # Fallback: use relative path from registry directory
            try:
                relative_path = record.artifact_path.relative_to(self._storage_dir)
            except ValueError:
                # Last resort: just store the artifact name and expect it in same directory
                relative_path = Path(record.artifact_path.name)
        
        payload = {
            "model_name": record.model_name,
            "version": record.version,
            "artifact_path": str(relative_path),
            "metrics": record.metrics,
            "created_at": record.created_at,
            "status": record.status,
            "metadata": record.metadata,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_existing_registry(self) -> None:
        """Load persisted registry entries from disk into memory."""
        for registry_file in sorted(self._storage_dir.glob("*_registry.json")):
            try:
                payload = json.loads(registry_file.read_text(encoding="utf-8"))
            except ValueError:
                self._logger.warning("Invalid registry payload", file=str(registry_file))
                continue

            model_name = payload.get("model_name")
            version = payload.get("version")
            artifact_path_str = payload.get("artifact_path")
            metrics = payload.get("metrics", {})
            created_at = payload.get("created_at", "")
            status = payload.get("status", "")
            metadata = payload.get("metadata", {})
            
            if not model_name or not isinstance(version, int) or not artifact_path_str:
                self._logger.warning("Skipping malformed registry entry", file=str(registry_file))
                continue

            # Resolve the artifact path - it may be relative or absolute
            artifact_path = Path(artifact_path_str)
            if not artifact_path.is_absolute():
                # If relative, resolve it relative to the registry directory's parent structure
                # Assumes standard structure: ml/data/training/model_registry/ -> ml/artifacts/models/
                artifact_path = (self._storage_dir.parent.parent / artifact_path).resolve()
                if not artifact_path.exists():
                    # Try relative to registry directory directly
                    artifact_path = (self._storage_dir / artifact_path_str).resolve()
            
            registered_model = RegisteredModel(
                model_name=model_name,
                version=version,
                artifact_path=artifact_path,
                metrics=dict(metrics),
                created_at=created_at,
                status=status,
                metadata=dict(metadata),
            )
            
            self._models[(model_name, version)] = registered_model

    def set_production(self, model_name: str, version: int) -> RegisteredModel:
        """Promote a model to Production and archive any previous Production model."""
        if not model_name:
            raise ValueError("model_name cannot be empty")

        target = self.get_model(model_name, version)
        for other in self._models.values():
            if other.model_name == model_name and other.status == "Production" and other.version != version:
                other.status = "Archived"
                self._persist_registration(other)
                self._logger.info("Model archived", model_name=other.model_name, version=other.version)

        target.status = "Production"
        self._persist_registration(target)
        self._logger.info("Production switched", model_name=model_name, version=version)
        return target
