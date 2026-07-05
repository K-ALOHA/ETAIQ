"""Production monitoring utilities for ETAIQ model predictions."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .logging_config import TrainingLogger


@dataclass
class MonitoringRecord:
    """Summary statistics for a batch of predictions."""

    timestamp: str
    model_name: str
    prediction_count: int
    mean_prediction: float
    std_prediction: float
    min_prediction: float
    max_prediction: float
    missing_inputs: int
    out_of_range_inputs: int


class MonitoringEngine:
    """Collect prediction summaries and export them for monitoring workflows."""

    def __init__(self, logger: TrainingLogger | None = None, *, load_existing_records: bool = False) -> None:
        self._logger = logger or TrainingLogger(name="training.monitoring")
        self._storage_dir = Path("ml") / "data" / "training" / "monitoring"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._storage_file = self._storage_dir / "monitoring_records.json"
        self._records: list[MonitoringRecord] = []
        if load_existing_records:
            self._load_existing_records()

    def record_predictions(
        self,
        model_name: str,
        predictions: Any,
        missing_inputs: int = 0,
        out_of_range_inputs: int = 0,
    ) -> MonitoringRecord:
        """Create a monitoring record from a batch of predictions."""
        if not model_name:
            raise ValueError("model_name cannot be empty")

        prediction_array = np.asarray(predictions, dtype=float).reshape(-1)
        if prediction_array.size == 0:
            raise ValueError("predictions cannot be empty")

        record = MonitoringRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
            prediction_count=int(prediction_array.size),
            mean_prediction=float(prediction_array.mean()),
            std_prediction=float(prediction_array.std()),
            min_prediction=float(prediction_array.min()),
            max_prediction=float(prediction_array.max()),
            missing_inputs=int(missing_inputs),
            out_of_range_inputs=int(out_of_range_inputs),
        )
        self._records.append(record)
        self._persist_records()
        self._logger.info("Monitoring record created", model_name=model_name, prediction_count=record.prediction_count)
        return record

    def _persist_records(self) -> None:
        payload = [asdict(record) for record in self._records]
        self._storage_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._logger.info("Monitoring records persisted", path=str(self._storage_file))

    def _load_existing_records(self) -> None:
        if not self._storage_file.exists():
            return
        try:
            payload = json.loads(self._storage_file.read_text(encoding="utf-8"))
            self._records = [MonitoringRecord(**record) for record in payload if isinstance(record, dict)]
        except Exception:
            self._logger.warning("Failed to load existing monitoring records", path=str(self._storage_file))

    def get_latest(self) -> MonitoringRecord | None:
        """Return the latest monitoring record, if one exists."""
        if not self._records:
            return None
        return self._records[-1]

    def list_records(self) -> list[MonitoringRecord]:
        """Return all monitoring records."""
        return list(self._records)

    def export_csv(self, output_path: str | Path) -> Path:
        """Export monitoring records to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "timestamp",
            "model_name",
            "prediction_count",
            "mean_prediction",
            "std_prediction",
            "min_prediction",
            "max_prediction",
            "missing_inputs",
            "out_of_range_inputs",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._records:
                writer.writerow(asdict(record))

        self._logger.info("Monitoring exported", output_path=str(path), format="csv")
        return path

    def export_json(self, output_path: str | Path) -> Path:
        """Export monitoring records to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(record) for record in self._records], indent=2), encoding="utf-8")
        self._logger.info("Monitoring exported", output_path=str(path), format="json")
        return path
