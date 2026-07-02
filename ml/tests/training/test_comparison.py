"""Unit tests for the model comparison engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.training.comparison import ModelComparisonEngine, ModelComparisonResult
from ml.training.evaluation import EvaluationResult


@pytest.fixture
def engine() -> ModelComparisonEngine:
    """Create a comparison engine for tests."""
    return ModelComparisonEngine()


def make_result(model_name: str, mae: float, rmse: float, r2: float, mape: float) -> EvaluationResult:
    """Create a simple evaluation result for comparison tests."""
    return EvaluationResult(
        model_name=model_name,
        predictions=[0.0],
        mae=mae,
        rmse=rmse,
        r2=r2,
        mape=mape,
        evaluation_time_seconds=0.1,
        evaluation_rows=1,
    )


def test_compare_three_models(engine: ModelComparisonEngine) -> None:
    """Three models should be ranked in a leaderboard."""
    results = [
        make_result("ModelA", 2.0, 3.0, 0.7, 8.0),
        make_result("ModelB", 1.0, 2.0, 0.8, 5.0),
        make_result("ModelC", 3.0, 4.0, 0.6, 10.0),
    ]

    comparison = engine.compare(results, ranking_metric="mae")

    assert isinstance(comparison, ModelComparisonResult)
    assert comparison.best_model_name == "ModelB"
    assert comparison.number_of_models == 3
    assert comparison.leaderboard[0]["model_name"] == "ModelB"


def test_mae_ranking(engine: ModelComparisonEngine) -> None:
    """MAE ranking should choose the lowest MAE."""
    results = [
        make_result("ModelA", 4.0, 3.0, 0.7, 8.0),
        make_result("ModelB", 1.0, 2.0, 0.8, 5.0),
    ]

    comparison = engine.compare(results, ranking_metric="mae")

    assert comparison.best_model_name == "ModelB"


def test_rmse_ranking(engine: ModelComparisonEngine) -> None:
    """RMSE ranking should choose the lowest RMSE."""
    results = [
        make_result("ModelA", 4.0, 5.0, 0.7, 8.0),
        make_result("ModelB", 2.0, 1.0, 0.9, 4.0),
    ]

    comparison = engine.compare(results, ranking_metric="rmse")

    assert comparison.best_model_name == "ModelB"


def test_mape_ranking(engine: ModelComparisonEngine) -> None:
    """MAPE ranking should choose the lowest MAPE."""
    results = [
        make_result("ModelA", 4.0, 5.0, 0.7, 10.0),
        make_result("ModelB", 2.0, 1.0, 0.9, 2.0),
    ]

    comparison = engine.compare(results, ranking_metric="mape")

    assert comparison.best_model_name == "ModelB"


def test_r2_ranking(engine: ModelComparisonEngine) -> None:
    """R² ranking should choose the highest R²."""
    results = [
        make_result("ModelA", 4.0, 5.0, 0.7, 10.0),
        make_result("ModelB", 2.0, 1.0, 0.9, 2.0),
    ]

    comparison = engine.compare(results, ranking_metric="r2")

    assert comparison.best_model_name == "ModelB"


def test_invalid_metric_raises_value_error(engine: ModelComparisonEngine) -> None:
    """Unsupported metrics should be rejected."""
    with pytest.raises(ValueError, match="Unsupported ranking metric"):
        engine.compare([make_result("ModelA", 1.0, 1.0, 0.8, 2.0)], ranking_metric="accuracy")


def test_duplicate_model_names_raises_value_error(engine: ModelComparisonEngine) -> None:
    """Duplicate model names should be rejected."""
    with pytest.raises(ValueError, match="Duplicate model names"):
        engine.compare(
            [
                make_result("ModelA", 1.0, 1.0, 0.8, 2.0),
                make_result("ModelA", 2.0, 2.0, 0.7, 3.0),
            ],
            ranking_metric="mae",
        )


def test_empty_list_raises_value_error(engine: ModelComparisonEngine) -> None:
    """An empty list should be rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        engine.compare([], ranking_metric="mae")


def test_export(engine: ModelComparisonEngine, tmp_path: Path) -> None:
    """The leaderboard should export to CSV."""
    engine._export_path = tmp_path / "model_leaderboard.csv"
    results = [
        make_result("ModelA", 2.0, 3.0, 0.7, 8.0),
        make_result("ModelB", 1.0, 2.0, 0.8, 5.0),
    ]

    comparison = engine.compare(results, ranking_metric="mae")
    export_path = engine.export_leaderboard(comparison)

    assert export_path.exists()
    content = export_path.read_text(encoding="utf-8")
    assert "Model" in content
    assert "ModelB" in content
    assert "ModelA" in content


def test_returned_dataclass(engine: ModelComparisonEngine) -> None:
    """The engine should return the expected dataclass."""
    results = [
        make_result("ModelA", 2.0, 3.0, 0.7, 8.0),
        make_result("ModelB", 1.0, 2.0, 0.8, 5.0),
    ]

    comparison = engine.compare(results, ranking_metric="mae")

    assert isinstance(comparison, ModelComparisonResult)
    assert comparison.ranking_metric == "mae"
    assert comparison.number_of_models == 2
