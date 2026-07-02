"""Production experiment tracking for ETAIQ training runs."""

from __future__ import annotations

import csv
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_config import TrainingLogger


@dataclass
class ExperimentRecord:
    """A single experiment execution record."""

    experiment_id: str
    timestamp: str
    model_name: str
    dataset_version: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    training_time_seconds: float
    model_version: int


class ExperimentTrackingEngine:
    """Store, retrieve, and export experiment metadata in memory."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.experiment_tracking")
        self._experiments: dict[str, ExperimentRecord] = {}

    def log_experiment(
        self,
        model_name: str,
        dataset_version: str,
        hyperparameters: dict[str, Any],
        metrics: dict[str, float],
        training_time_seconds: float,
        model_version: int,
        experiment_id: str | None = None,
    ) -> ExperimentRecord:
        """Create a new experiment record and store it in memory."""
        if not model_name:
            raise ValueError("model_name cannot be empty")
        if not dataset_version:
            raise ValueError("dataset_version cannot be empty")
        if training_time_seconds < 0:
            raise ValueError("training_time_seconds cannot be negative")

        resolved_experiment_id = experiment_id or self._generate_experiment_id()
        if resolved_experiment_id in self._experiments:
            raise ValueError(f"duplicate experiment ID: {resolved_experiment_id}")

        record = ExperimentRecord(
            experiment_id=resolved_experiment_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
            dataset_version=dataset_version,
            hyperparameters=dict(hyperparameters),
            metrics=dict(metrics),
            training_time_seconds=float(training_time_seconds),
            model_version=model_version,
        )
        self._experiments[record.experiment_id] = record
        self._logger.info("Experiment created", experiment_id=record.experiment_id, model_name=model_name)
        return record

    def list_experiments(self) -> list[ExperimentRecord]:
        """Return all stored experiment records."""
        return list(self._experiments.values())

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Return an experiment by ID."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        return self._experiments[experiment_id]

    def export_csv(self, output_path: str | Path) -> Path:
        """Export all experiments to a CSV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "experiment_id",
            "timestamp",
            "model_name",
            "dataset_version",
            "hyperparameters",
            "metrics",
            "training_time_seconds",
            "model_version",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.list_experiments():
                writer.writerow(
                    {
                        "experiment_id": record.experiment_id,
                        "timestamp": record.timestamp,
                        "model_name": record.model_name,
                        "dataset_version": record.dataset_version,
                        "hyperparameters": json.dumps(record.hyperparameters),
                        "metrics": json.dumps(record.metrics),
                        "training_time_seconds": record.training_time_seconds,
                        "model_version": record.model_version,
                    }
                )

        self._logger.info("Experiment exported", output_path=str(path), format="csv")
        return path

    def export_json(self, output_path: str | Path) -> Path:
        """Export all experiments to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(record) for record in self.list_experiments()]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._logger.info("Experiment exported", output_path=str(path), format="json")
        return path

    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment identifier."""
        return f"exp-{uuid.uuid4().hex[:8]}"
