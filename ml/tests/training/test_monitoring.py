"""Unit tests for production monitoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.training.monitoring import MonitoringEngine, MonitoringRecord


def test_record_predictions_and_retrieve_latest() -> None:
    """Monitoring should store prediction summaries and return the latest record."""
    engine = MonitoringEngine()
    predictions = np.array([1.0, 2.0, 3.0])

    record = engine.record_predictions("LinearRegression", predictions, missing_inputs=0, out_of_range_inputs=0)

    assert isinstance(record, MonitoringRecord)
    assert record.model_name == "LinearRegression"
    assert record.prediction_count == 3
    assert record.mean_prediction == 2.0
    assert engine.get_latest().model_name == "LinearRegression"
    assert len(engine.list_records()) == 1


def test_export_monitoring_records(tmp_path: Path) -> None:
    """Monitoring export methods should write CSV and JSON files."""
    engine = MonitoringEngine()
    engine.record_predictions("LinearRegression", np.array([1.0, 2.0, 3.0]))

    csv_path = engine.export_csv(tmp_path / "monitoring.csv")
    json_path = engine.export_json(tmp_path / "monitoring.json")

    assert csv_path.exists()
    assert json_path.exists()
