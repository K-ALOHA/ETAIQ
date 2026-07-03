"""Unit tests for the end-to-end training pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from ml.training.pipeline import TrainingPipelineEngine, TrainingPipelineResult


@pytest.fixture
def engine() -> TrainingPipelineEngine:
    """Create a training pipeline engine for tests."""
    return TrainingPipelineEngine()


def test_successful_pipeline(engine: TrainingPipelineEngine) -> None:
    """The pipeline should train, evaluate, compare, select, and save a model."""
    X_train = np.arange(30).reshape(-1, 1)
    X_test = np.arange(30, 40).reshape(-1, 1)
    y_train = np.arange(30, dtype=float)
    y_test = np.arange(30, 40, dtype=float)

    result = engine.run(X_train, X_test, y_train, y_test)

    assert isinstance(result, TrainingPipelineResult)
    assert len(result.trained_models) == 4
    assert len(result.evaluation_results) == 4
    assert result.comparison_result is not None
    assert result.best_model is not None
    assert result.saved_model is not None
    assert result.pipeline_time_seconds >= 0.0


def test_multiple_models_trained(engine: TrainingPipelineEngine) -> None:
    """The pipeline should train each registered model."""
    X_train = np.arange(24).reshape(-1, 1)
    X_test = np.arange(24, 32).reshape(-1, 1)
    y_train = np.arange(24, dtype=float)
    y_test = np.arange(24, 32, dtype=float)

    result = engine.run(X_train, X_test, y_train, y_test)

    assert len(result.trained_models) == 4


def test_evaluation_generated(engine: TrainingPipelineEngine) -> None:
    """The pipeline should produce evaluation results for each model."""
    X_train = np.arange(20).reshape(-1, 1)
    X_test = np.arange(20, 28).reshape(-1, 1)
    y_train = np.arange(20, dtype=float)
    y_test = np.arange(20, 28, dtype=float)

    result = engine.run(X_train, X_test, y_train, y_test)

    assert len(result.evaluation_results) == 4


def test_comparison_generated(engine: TrainingPipelineEngine) -> None:
    """The pipeline should produce a comparison result."""
    X_train = np.arange(20).reshape(-1, 1)
    X_test = np.arange(20, 28).reshape(-1, 1)
    y_train = np.arange(20, dtype=float)
    y_test = np.arange(20, 28, dtype=float)

    result = engine.run(X_train, X_test, y_train, y_test)

    assert result.comparison_result.number_of_models == 4


def test_best_model_selected(engine: TrainingPipelineEngine) -> None:
    """The pipeline should select a best model."""
    X_train = np.arange(20).reshape(-1, 1)
    X_test = np.arange(20, 28).reshape(-1, 1)
    y_train = np.arange(20, dtype=float)
    y_test = np.arange(20, 28, dtype=float)

    result = engine.run(X_train, X_test, y_train, y_test)

    assert result.best_model.model_name in {"LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor", "XGBRegressor"}


def test_model_persisted(engine: TrainingPipelineEngine) -> None:
    """The pipeline should save the best model."""
    X_train = np.arange(20).reshape(-1, 1)
    X_test = np.arange(20, 28).reshape(-1, 1)
    y_train = np.arange(20, dtype=float)
    y_test = np.arange(20, 28, dtype=float)

    result = engine.run(X_train, X_test, y_train, y_test)

    assert result.saved_model.model_path.exists()
    assert result.saved_model.metadata_path.exists()


def test_pipeline_timing(engine: TrainingPipelineEngine) -> None:
    """The pipeline should measure the elapsed time."""
    X_train = np.arange(20).reshape(-1, 1)
    X_test = np.arange(20, 28).reshape(-1, 1)
    y_train = np.arange(20, dtype=float)
    y_test = np.arange(20, 28, dtype=float)

    result = engine.run(X_train, X_test, y_train, y_test)

    assert result.pipeline_time_seconds >= 0.0


def test_invalid_input(engine: TrainingPipelineEngine) -> None:
    """Empty input should be rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        engine.run(np.array([]), np.array([]), np.array([]), np.array([]))
