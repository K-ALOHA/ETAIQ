"""Unit tests for the end-to-end prediction pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from ml.training.persistence import ModelPersistenceEngine
from ml.training.prediction_pipeline import PredictionPipelineEngine, PredictionPipelineResult


@pytest.fixture
def engine() -> PredictionPipelineEngine:
    """Create a prediction pipeline engine for tests."""
    return PredictionPipelineEngine()


@pytest.fixture
def saved_model_path(tmp_path) -> str:
    """Persist a simple regression model for prediction pipeline tests."""
    persistence_engine = ModelPersistenceEngine()
    model = LinearRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    result = persistence_engine.save_model(model, "LinearRegression")
    return str(result.model_path)


def test_successful_prediction(engine: PredictionPipelineEngine, saved_model_path: str) -> None:
    """The pipeline should produce predictions for valid input."""
    result = engine.predict(saved_model_path, np.array([[3.0], [4.0]]))

    assert isinstance(result, PredictionPipelineResult)
    assert result.number_of_rows == 2
    assert result.model_name == "LinearRegression"
    assert result.model_version >= 1
    assert result.predictions.shape == (2,)
    assert result.pipeline_time_seconds >= 0.0


def test_dataframe_input(engine: PredictionPipelineEngine, saved_model_path: str) -> None:
    """The pipeline should accept dataframe input."""
    frame = pd.DataFrame({"feature": [3.0, 4.0]})
    result = engine.predict(saved_model_path, frame)

    assert result.number_of_rows == 2
    assert result.predictions.shape == (2,)


def test_invalid_model_path(engine: PredictionPipelineEngine) -> None:
    """A missing model file should be rejected."""
    with pytest.raises(ValueError, match="Model file missing"):
        engine.predict("/tmp/does-not-exist.joblib", np.array([[1.0]]))


def test_empty_dataframe(engine: PredictionPipelineEngine, saved_model_path: str) -> None:
    """Empty dataframe input should be rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        engine.predict(saved_model_path, pd.DataFrame(columns=["feature"]))


def test_none_input(engine: PredictionPipelineEngine, saved_model_path: str) -> None:
    """None input should be rejected."""
    with pytest.raises(ValueError, match="cannot be None"):
        engine.predict(saved_model_path, None)


def test_returned_dataclass(engine: PredictionPipelineEngine, saved_model_path: str) -> None:
    """The engine should return the expected dataclass."""
    result = engine.predict(saved_model_path, np.array([[3.0]]))

    assert isinstance(result, PredictionPipelineResult)
    assert isinstance(result.pipeline_time_seconds, float)
