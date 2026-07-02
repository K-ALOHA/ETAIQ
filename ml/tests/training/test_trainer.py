"""Unit tests for the training engine."""

from __future__ import annotations

import numpy as np
import pytest

from ml.training.models import ModelDefinition
from ml.training.registry import ModelRegistry
from ml.training.trainer import TrainingEngine, TrainingResult


@pytest.fixture
def engine() -> TrainingEngine:
    """Create a training engine instance for tests."""
    return TrainingEngine(registry=ModelRegistry())


def test_successful_training(engine: TrainingEngine) -> None:
    """Training should fit a model and return a result object."""
    X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([1.0, 2.0, 3.0, 4.0])

    result = engine.train("LinearRegression", X_train, y_train)

    assert isinstance(result, TrainingResult)
    assert result.model_name == "LinearRegression"
    assert result.training_rows == 4
    assert result.training_features == 1
    assert result.training_time_seconds >= 0
    assert result.trained_model is not None


def test_training_uses_registry_integration(engine: TrainingEngine) -> None:
    """The engine should delegate model creation to the registry."""
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([0.0, 1.0, 2.0])

    result = engine.train("LinearRegression", X_train, y_train)

    assert result.model_name == "LinearRegression"
    assert result.trained_model.__class__.__name__ == "LinearRegression"


def test_invalid_model_raises_value_error(engine: TrainingEngine) -> None:
    """Unknown models should raise a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown model name"):
        engine.train("NotARealModel", np.array([[1.0]]), np.array([1.0]))


def test_empty_data_raises_value_error(engine: TrainingEngine) -> None:
    """Empty datasets should be rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        engine.train("LinearRegression", np.array([]), np.array([]))


def test_mismatched_lengths_raise_value_error(engine: TrainingEngine) -> None:
    """Mismatched feature and target lengths should be rejected."""
    with pytest.raises(ValueError, match="must have the same length"):
        engine.train("LinearRegression", np.array([[1.0], [2.0]]), np.array([1.0]))


def test_training_result_contains_expected_fields(engine: TrainingEngine) -> None:
    """The returned result should expose the expected metadata."""
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_train = np.array([3.0, 5.0, 7.0])

    result = engine.train("LinearRegression", X_train, y_train)

    assert result.model_name == "LinearRegression"
    assert result.training_rows == 3
    assert result.training_features == 2
    assert isinstance(result.training_time_seconds, float)


def test_fitted_model_can_predict(engine: TrainingEngine) -> None:
    """A fitted model should be usable for prediction."""
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0.0, 1.0, 2.0, 3.0])

    result = engine.train("LinearRegression", X_train, y_train)
    prediction = result.trained_model.predict(np.array([[4.0]]))

    assert prediction.shape == (1,)
    assert float(prediction[0]) == pytest.approx(4.0)
