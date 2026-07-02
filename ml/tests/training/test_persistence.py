"""Unit tests for the model persistence engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.linear_model import LinearRegression

from ml.training.persistence import ModelPersistenceEngine, PersistenceResult


@pytest.fixture
def engine() -> ModelPersistenceEngine:
    """Create a persistence engine for tests."""
    return ModelPersistenceEngine()


def test_save_model(engine: ModelPersistenceEngine) -> None:
    """A trained model should be saved with versioned files."""
    model = LinearRegression()
    result = engine.save_model(model, "LinearRegression")

    assert isinstance(result, PersistenceResult)
    assert result.model_name == "LinearRegression"
    assert result.model_path.exists()
    assert result.metadata_path.exists()
    assert result.version >= 1


def test_load_model(engine: ModelPersistenceEngine) -> None:
    """A saved model should be loadable from disk."""
    model = LinearRegression()
    saved = engine.save_model(model, "LinearRegression")

    loaded_model = engine.load_model(saved.model_path)
    assert isinstance(loaded_model, LinearRegression)


def test_metadata_correctness(engine: ModelPersistenceEngine) -> None:
    """Metadata should include the expected fields."""
    model = LinearRegression()
    saved = engine.save_model(model, "LinearRegression", metadata={"owner": "tests"})

    payload = json.loads(saved.metadata_path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "LinearRegression"
    assert payload["version"] >= 1
    assert payload["framework"] == "scikit-learn"
    assert payload["custom_metadata"]["owner"] == "tests"


def test_automatic_version_increment(engine: ModelPersistenceEngine) -> None:
    """Saving the same model name twice should increment the version."""
    engine.save_model(LinearRegression(), "RandomForest")
    second_result = engine.save_model(LinearRegression(), "RandomForest")

    assert second_result.version > 1
    assert second_result.model_path.name.startswith("RandomForest_v")
    assert second_result.model_path.name.endswith(".joblib")


def test_none_model_raises_value_error(engine: ModelPersistenceEngine) -> None:
    """A None model should be rejected."""
    with pytest.raises(ValueError, match="model cannot be None"):
        engine.save_model(None, "LinearRegression")


def test_empty_model_name_raises_value_error(engine: ModelPersistenceEngine) -> None:
    """An empty model name should be rejected."""
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        engine.save_model(LinearRegression(), "")


def test_missing_file_raises_value_error(engine: ModelPersistenceEngine) -> None:
    """Missing model files should be rejected on load."""
    with pytest.raises(ValueError, match="Model file missing"):
        engine.load_model("/tmp/does-not-exist.joblib")


def test_returned_dataclass(engine: ModelPersistenceEngine) -> None:
    """The engine should return the expected dataclass."""
    result = engine.save_model(LinearRegression(), "LinearRegression")

    assert isinstance(result, PersistenceResult)
    assert result.file_size_bytes > 0
