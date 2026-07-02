"""Unit tests for the inference engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from ml.training.inference import InferenceEngine, InferenceResult
from ml.training.persistence import ModelPersistenceEngine


@pytest.fixture
def engine() -> InferenceEngine:
    """Create an inference engine for tests."""
    return InferenceEngine()


@pytest.fixture
def saved_model_path(tmp_path) -> str:
    """Persist a simple model for inference tests."""
    persistence_engine = ModelPersistenceEngine()
    model = LinearRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    result = persistence_engine.save_model(model, "LinearRegression")
    return str(result.model_path)


def test_successful_prediction(engine: InferenceEngine, saved_model_path: str) -> None:
    """Single prediction should return an inference result."""
    result = engine.predict(saved_model_path, np.array([[3.0]]))

    assert isinstance(result, InferenceResult)
    assert result.model_name == "LinearRegression"
    assert result.input_rows == 1
    assert result.model_version >= 1


def test_batch_prediction(engine: InferenceEngine, saved_model_path: str) -> None:
    """Batch prediction should return a numpy array of predictions."""
    predictions = engine.batch_predict(saved_model_path, np.array([[3.0], [4.0]]))

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)


def test_invalid_model_path(engine: InferenceEngine) -> None:
    """A missing model file should be rejected."""
    with pytest.raises(ValueError, match="Model file missing"):
        engine.predict("/tmp/does-not-exist.joblib", np.array([[1.0]]))


def test_empty_dataframe(engine: InferenceEngine, saved_model_path: str) -> None:
    """Empty dataframe input should be rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        engine.predict(saved_model_path, pd.DataFrame(columns=["x"]))


def test_none_input(engine: InferenceEngine, saved_model_path: str) -> None:
    """None input should be rejected."""
    with pytest.raises(ValueError, match="cannot be None"):
        engine.predict(saved_model_path, None)


def test_prediction_shape(engine: InferenceEngine, saved_model_path: str) -> None:
    """Single prediction should return a scalar-like prediction object."""
    result = engine.predict(saved_model_path, np.array([[3.0]]))

    assert np.asarray(result.prediction).shape == ()


def test_returned_dataclass(engine: InferenceEngine, saved_model_path: str) -> None:
    """The engine should return the expected dataclass."""
    result = engine.predict(saved_model_path, np.array([[3.0]]))

    assert isinstance(result, InferenceResult)
    assert isinstance(result.prediction_time_seconds, float)


def test_prediction_timing(engine: InferenceEngine, saved_model_path: str) -> None:
    """Prediction timing should be captured."""
    result = engine.predict(saved_model_path, np.array([[3.0]]))

    assert result.prediction_time_seconds >= 0.0
