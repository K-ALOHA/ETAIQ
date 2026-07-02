"""Unit tests for the cross-validation engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.training.cross_validation import CrossValidationEngine, CrossValidationResult


@pytest.fixture
def engine() -> CrossValidationEngine:
    """Create a cross-validation engine for tests."""
    return CrossValidationEngine()


def test_successful_5_fold_validation(engine: CrossValidationEngine) -> None:
    """Five-fold cross-validation should produce a result object with five folds."""
    X = np.arange(50).reshape(-1, 1)
    y = np.arange(50, dtype=float)

    result = engine.cross_validate("LinearRegression", X, y, n_splits=5)

    assert isinstance(result, CrossValidationResult)
    assert result.number_of_folds == 5
    assert len(result.fold_results) == 5
    assert result.model_name == "LinearRegression"


def test_3_fold_validation(engine: CrossValidationEngine) -> None:
    """Three-fold cross-validation should work for smaller datasets."""
    X = np.arange(30).reshape(-1, 1)
    y = np.arange(30, dtype=float)

    result = engine.cross_validate("LinearRegression", X, y, n_splits=3)

    assert result.number_of_folds == 3
    assert len(result.fold_results) == 3


def test_invalid_model_raises_value_error(engine: CrossValidationEngine) -> None:
    """Unknown models should raise a clear error."""
    with pytest.raises(ValueError, match="Unknown model name"):
        engine.cross_validate("NotARealModel", np.array([[1.0]]), np.array([1.0]), n_splits=2)


def test_empty_data_raises_value_error(engine: CrossValidationEngine) -> None:
    """Empty datasets should be rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        engine.cross_validate("LinearRegression", np.array([]), np.array([]), n_splits=2)


def test_length_mismatch_raises_value_error(engine: CrossValidationEngine) -> None:
    """Mismatched feature and target lengths should be rejected."""
    with pytest.raises(ValueError, match="same length"):
        engine.cross_validate("LinearRegression", np.array([[1.0], [2.0]]), np.array([1.0]), n_splits=2)


def test_invalid_n_splits_raises_value_error(engine: CrossValidationEngine) -> None:
    """Invalid fold counts should be rejected."""
    with pytest.raises(ValueError, match="at least 2"):
        engine.cross_validate("LinearRegression", np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]), n_splits=1)

    with pytest.raises(ValueError, match="cannot exceed"):
        engine.cross_validate("LinearRegression", np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]), n_splits=4)


def test_aggregation_correctness(engine: CrossValidationEngine) -> None:
    """Aggregated metrics should be computed from fold results."""
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20, dtype=float)

    result = engine.cross_validate("LinearRegression", X, y, n_splits=2)

    assert isinstance(result.mean_mae, float)
    assert isinstance(result.std_mae, float)
    assert isinstance(result.mean_rmse, float)
    assert isinstance(result.std_rmse, float)
    assert isinstance(result.mean_r2, float)
    assert isinstance(result.std_r2, float)
    assert isinstance(result.mean_mape, float)
    assert isinstance(result.std_mape, float)
    assert result.total_training_time_seconds >= 0.0


def test_export_correctness(engine: CrossValidationEngine, tmp_path: Path) -> None:
    """Cross-validation results should be exportable to CSV."""
    engine._export_path = tmp_path / "cross_validation_results.csv"
    X = np.arange(20).reshape(-1, 1)
    y = np.arange(20, dtype=float)

    result = engine.cross_validate("LinearRegression", X, y, n_splits=2)
    export_path = engine.export_results(result)

    assert export_path.exists()
    assert export_path.read_text(encoding="utf-8").count("fold") == 1


def test_returned_dataclass(engine: CrossValidationEngine) -> None:
    """The cross-validation engine should return the expected dataclass."""
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10, dtype=float)

    result = engine.cross_validate("LinearRegression", X, y, n_splits=2)

    assert isinstance(result, CrossValidationResult)
    assert result.number_of_folds == 2
    assert len(result.fold_results) == 2
