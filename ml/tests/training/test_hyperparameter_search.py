"""Unit tests for the hyperparameter search engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.training.hyperparameter_search import HyperparameterSearchEngine, HyperparameterSearchResult


@pytest.fixture
def engine() -> HyperparameterSearchEngine:
    """Create a hyperparameter search engine for tests."""
    return HyperparameterSearchEngine()


def test_successful_search(engine: HyperparameterSearchEngine) -> None:
    """A simple search should return a populated result object."""
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20, dtype=float)
    param_grid = {"fit_intercept": [True, False]}

    result = engine.search("LinearRegression", X, y, param_grid=param_grid, cv=2)

    assert isinstance(result, HyperparameterSearchResult)
    assert result.model_name == "LinearRegression"
    assert result.number_of_configurations == 2
    assert isinstance(result.best_parameters, dict)
    assert isinstance(result.best_score, float)
    assert result.best_model is not None


def test_random_forest_search(engine: HyperparameterSearchEngine) -> None:
    """Random forest should support a parameter grid search."""
    X = np.arange(30).reshape(-1, 1)
    y = np.arange(30, dtype=float)
    param_grid = {"n_estimators": [5, 10]}

    result = engine.search("RandomForestRegressor", X, y, param_grid=param_grid, cv=2)

    assert result.model_name == "RandomForestRegressor"
    assert result.number_of_configurations == 2


def test_gradient_boosting_search(engine: HyperparameterSearchEngine) -> None:
    """Gradient boosting should support a parameter grid search."""
    X = np.arange(30).reshape(-1, 1)
    y = np.arange(30, dtype=float)
    param_grid = {"n_estimators": [5, 10]}

    result = engine.search("GradientBoostingRegressor", X, y, param_grid=param_grid, cv=2)

    assert result.model_name == "GradientBoostingRegressor"
    assert result.number_of_configurations == 2


def test_linear_regression_search(engine: HyperparameterSearchEngine) -> None:
    """Linear regression should work with a parameter grid."""
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20, dtype=float)
    param_grid = {"fit_intercept": [True]}

    result = engine.search("LinearRegression", X, y, param_grid=param_grid, cv=2)

    assert result.model_name == "LinearRegression"
    assert result.number_of_configurations == 1


def test_empty_dataset_raises_value_error(engine: HyperparameterSearchEngine) -> None:
    """Empty data should be rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        engine.search("LinearRegression", np.array([]), np.array([]), param_grid={"fit_intercept": [True]}, cv=2)


def test_length_mismatch_raises_value_error(engine: HyperparameterSearchEngine) -> None:
    """Mismatched feature and target lengths should be rejected."""
    with pytest.raises(ValueError, match="same length"):
        engine.search("LinearRegression", np.array([[1.0], [2.0]]), np.array([1.0]), param_grid={"fit_intercept": [True]}, cv=2)


def test_invalid_model_raises_value_error(engine: HyperparameterSearchEngine) -> None:
    """Unknown models should raise a clear error."""
    with pytest.raises(ValueError, match="Unknown model name"):
        engine.search("NotARealModel", np.array([[1.0]]), np.array([1.0]), param_grid={"fit_intercept": [True]}, cv=2)


def test_invalid_parameter_grid_raises_value_error(engine: HyperparameterSearchEngine) -> None:
    """An empty parameter grid should be rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        engine.search("LinearRegression", np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), param_grid={}, cv=2)


def test_returned_dataclass(engine: HyperparameterSearchEngine) -> None:
    """The engine should return the expected dataclass."""
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10, dtype=float)
    param_grid = {"fit_intercept": [True]}

    result = engine.search("LinearRegression", X, y, param_grid=param_grid, cv=2)

    assert isinstance(result, HyperparameterSearchResult)
    assert result.number_of_configurations == 1


def test_export_correctness(engine: HyperparameterSearchEngine, tmp_path: Path) -> None:
    """Search results should be exportable to CSV."""
    engine._export_path = tmp_path / "hyperparameter_search_results.csv"
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10, dtype=float)
    param_grid = {"fit_intercept": [True, False]}

    result = engine.search("LinearRegression", X, y, param_grid=param_grid, cv=2)
    export_path = engine.export_results(result)

    assert export_path.exists()
    assert export_path.read_text(encoding="utf-8").count("model_name") == 1


def test_best_parameter_detection(engine: HyperparameterSearchEngine) -> None:
    """The engine should preserve the best parameters from GridSearchCV."""
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20, dtype=float)
    param_grid = {"fit_intercept": [True, False]}

    result = engine.search("LinearRegression", X, y, param_grid=param_grid, cv=2)

    assert result.best_parameters in [{"fit_intercept": True}, {"fit_intercept": False}]
