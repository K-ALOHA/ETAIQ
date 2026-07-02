"""Unit tests for production experiment tracking."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.training.experiment_tracking import ExperimentRecord, ExperimentTrackingEngine


def test_log_and_retrieve_experiment() -> None:
    """Experiments should be logged and retrievable by ID."""
    engine = ExperimentTrackingEngine()

    record = engine.log_experiment(
        model_name="LinearRegression",
        dataset_version="v1",
        hyperparameters={"alpha": 0.1},
        metrics={"mae": 0.5},
        training_time_seconds=1.23,
        model_version=2,
    )

    assert isinstance(record, ExperimentRecord)
    assert record.model_name == "LinearRegression"
    assert record.model_version == 2
    assert len(engine.list_experiments()) == 1
    assert engine.get_experiment(record.experiment_id).experiment_id == record.experiment_id


def test_duplicate_experiment_id_raises_value_error() -> None:
    """A duplicate experiment ID should be rejected."""
    engine = ExperimentTrackingEngine()
    engine.log_experiment(
        model_name="LinearRegression",
        dataset_version="v1",
        hyperparameters={},
        metrics={},
        training_time_seconds=1.0,
        model_version=1,
        experiment_id="exp-1",
    )

    with pytest.raises(ValueError, match="duplicate experiment"):
        engine.log_experiment(
            model_name="LinearRegression",
            dataset_version="v1",
            hyperparameters={},
            metrics={},
            training_time_seconds=1.0,
            model_version=1,
            experiment_id="exp-1",
        )


def test_export_experiments_to_csv_and_json(tmp_path: Path) -> None:
    """Experiment exports should produce both CSV and JSON files."""
    engine = ExperimentTrackingEngine()
    engine.log_experiment(
        model_name="LinearRegression",
        dataset_version="v1",
        hyperparameters={"alpha": 0.1},
        metrics={"mae": 0.5},
        training_time_seconds=1.23,
        model_version=2,
    )

    csv_path = engine.export_csv(tmp_path / "experiments.csv")
    json_path = engine.export_json(tmp_path / "experiments.json")

    assert csv_path.exists()
    assert json_path.exists()
    assert csv_path.suffix == ".csv"
    assert json_path.suffix == ".json"
