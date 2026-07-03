"""Unit tests for the training model registry."""

from __future__ import annotations

import pytest

from ml.training.registry import ModelRegistry


@pytest.fixture
def registry() -> ModelRegistry:
    """Create a registry instance for tests."""
    return ModelRegistry()


def test_registry_initialization(registry: ModelRegistry) -> None:
    """The registry should initialize with the expected built-in models."""
    assert registry is not None
    assert registry.list_models() == [
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "XGBRegressor",
    ]


def test_registered_model_count(registry: ModelRegistry) -> None:
    """Exactly three regression models should be registered."""
    assert len(registry.list_models()) == 4


def test_model_retrieval_returns_fresh_instance(registry: ModelRegistry) -> None:
    """Requesting a model should return a new sklearn instance each time."""
    first = registry.get_model("LinearRegression")
    second = registry.get_model("LinearRegression")

    assert first is not None
    assert second is not None
    assert type(first) is type(second)
    assert first is not second


def test_invalid_model_lookup_raises_clear_error(registry: ModelRegistry) -> None:
    """Unknown model names should raise a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown model name"):
        registry.get_model("NotARealModel")

    with pytest.raises(ValueError, match="Unknown model name"):
        registry.get_metadata("NotARealModel")


def test_metadata_validation(registry: ModelRegistry) -> None:
    """Metadata should contain the expected properties for each model."""
    metadata = registry.get_metadata("RandomForestRegressor")

    assert metadata["model_name"] == "RandomForestRegressor"
    assert metadata["task"] == "regression"
    assert metadata["needs_scaling"] is False
    assert metadata["supports_feature_importance"] is True
    assert isinstance(metadata["default_parameters"], dict)

    assert registry.has_model("GradientBoostingRegressor") is True
    assert registry.has_model("NotARealModel") is False
