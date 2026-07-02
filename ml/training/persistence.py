"""Model persistence engine for the ETAIQ production training pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn
import sys

from .config import DEFAULT_TRAINING_CONFIG
from .logging_config import TrainingLogger


@dataclass
class PersistenceResult:
    """Container for a completed model persistence operation."""

    model_name: str
    model_path: Path
    metadata_path: Path
    version: int
    saved_timestamp: str
    file_size_bytes: int


class ModelPersistenceEngine:
    """Persist and load trained models with versioned metadata."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.persistence")
        self._models_dir = DEFAULT_TRAINING_CONFIG.models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, trained_model: Any, model_name: str, metadata: dict[str, Any] | None = None) -> PersistenceResult:
        """Persist a trained model and emit versioned metadata."""
        self._validate_save_inputs(trained_model, model_name, metadata)

        self._logger.info("Model save started", model_name=model_name)

        version = self._next_version(model_name)
        timestamp = datetime.now(timezone.utc).isoformat()
        model_path = self._models_dir / f"{model_name}_v{version}.joblib"
        metadata_path = self._models_dir / f"{model_name}_v{version}.json"

        start_time = time.perf_counter()
        joblib.dump(trained_model, model_path)
        elapsed = time.perf_counter() - start_time

        metadata_payload = {
            "model_name": model_name,
            "version": version,
            "saved_timestamp": timestamp,
            "framework": "scikit-learn",
            "python_version": sys.version.split()[0],
            "sklearn_version": sklearn.__version__,
            "custom_metadata": metadata or {},
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

        self._logger.info(
            "Model saved",
            model_name=model_name,
            model_path=str(model_path),
            metadata_path=str(metadata_path),
            version=version,
            execution_time_seconds=elapsed,
        )
        self._logger.info("Metadata saved", model_name=model_name, metadata_path=str(metadata_path))

        return PersistenceResult(
            model_name=model_name,
            model_path=model_path,
            metadata_path=metadata_path,
            version=version,
            saved_timestamp=timestamp,
            file_size_bytes=model_path.stat().st_size,
        )

    def load_model(self, model_path: str | Path) -> Any:
        """Load a persisted model from disk."""
        path = Path(model_path)
        if not path.exists():
            raise ValueError(f"Model file missing: {path}")

        self._logger.info("Model loaded", model_path=str(path))
        return joblib.load(path)

    def _next_version(self, model_name: str) -> int:
        """Determine the next available version for a model name."""
        version = 1
        while (self._models_dir / f"{model_name}_v{version}.joblib").exists():
            version += 1
        return version

    def _validate_save_inputs(self, trained_model: Any, model_name: str, metadata: dict[str, Any] | None) -> None:
        """Validate the inputs before saving a model."""
        if trained_model is None:
            raise ValueError("model cannot be None")

        if not model_name or not str(model_name).strip():
            raise ValueError("model_name cannot be empty")

        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
