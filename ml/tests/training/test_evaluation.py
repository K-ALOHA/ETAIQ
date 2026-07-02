"""Unit tests for the evaluation engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.training.evaluation import EvaluationEngine, EvaluationResult


@pytest.fixture
def engine() -> EvaluationEngine:
    """Create an evaluation engine instance for tests."""
    return EvaluationEngine()


def test_successful_evaluation(engine: EvaluationEngine) -> None:
    """Evaluation should return a populated result object."""
    model = np.poly1d([1.0, 0.0])
    X_test = np.array([[1.0], [2.0], [3.0]])
    y_test = np.array([1.0, 2.0, 3.0])

    result = engine.evaluate(model, "poly1d", X_test, y_test)

    assert isinstance(result, EvaluationResult)
    assert result.model_name == "poly1d"
    assert result.evaluation_rows == 3
    assert result.predictions.shape == (3,)


def test_prediction_length(engine: EvaluationEngine) -> None:
    """Predictions should match the number of rows in the test set."""
    model = np.poly1d([1.0, 0.0])
    X_test = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_test = np.array([1.0, 2.0, 3.0, 4.0])

    result = engine.evaluate(model, "poly1d", X_test, y_test)

    assert len(result.predictions) == 4


def test_correct_metric_values(engine: EvaluationEngine) -> None:
    """Metrics should be computed with the expected values."""
    model = np.poly1d([1.0, 0.0])
    X_test = np.array([[1.0], [2.0], [3.0]])
    y_test = np.array([1.0, 2.0, 3.0])

    result = engine.evaluate(model, "poly1d", X_test, y_test)

    assert result.mae == pytest.approx(0.0)
    assert result.rmse == pytest.approx(0.0)
    assert result.r2 == pytest.approx(1.0)
    assert result.mape == pytest.approx(0.0)


def test_export_works(engine: EvaluationEngine, tmp_path: Path) -> None:
    """Metrics should be exportable to CSV."""
    engine._export_path = tmp_path / "evaluation_metrics.csv"
    model = np.poly1d([1.0, 0.0])
    X_test = np.array([[1.0], [2.0]])
    y_test = np.array([1.0, 2.0])

    result = engine.evaluate(model, "poly1d", X_test, y_test)
    export_path = engine.export_metrics(result)

    assert export_path.exists()
    assert export_path.read_text(encoding="utf-8").count("poly1d") == 1


def test_empty_dataset_raises_value_error(engine: EvaluationEngine) -> None:
    """Empty datasets should be rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        engine.evaluate(np.poly1d([1.0, 0.0]), "poly1d", np.array([]), np.array([]))


def test_length_mismatch_raises_value_error(engine: EvaluationEngine) -> None:
    """Mismatched lengths should be rejected."""
    with pytest.raises(ValueError, match="same length"):
        engine.evaluate(np.poly1d([1.0, 0.0]), "poly1d", np.array([[1.0], [2.0]]), np.array([1.0]))


def test_none_model_raises_value_error(engine: EvaluationEngine) -> None:
    """A missing model should be rejected."""
    with pytest.raises(ValueError, match="cannot be None"):
        engine.evaluate(None, "poly1d", np.array([[1.0]]), np.array([1.0]))


def test_returned_evaluation_result(engine: EvaluationEngine) -> None:
    """The returned result should expose the expected fields."""
    model = np.poly1d([1.0, 0.0])
    X_test = np.array([[1.0], [2.0], [3.0]])
    y_test = np.array([1.0, 2.0, 3.0])

    result = engine.evaluate(model, "poly1d", X_test, y_test)

    assert result.model_name == "poly1d"
    assert isinstance(result.evaluation_time_seconds, float)
    assert result.evaluation_rows == 3
