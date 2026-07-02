"""Unit tests for the best model selection engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.training.comparison import ModelComparisonResult
from ml.training.selection import BestModelSelectionEngine, BestModelResult


@pytest.fixture
def engine() -> BestModelSelectionEngine:
    """Create a best model selection engine for tests."""
    return BestModelSelectionEngine()


def make_comparison_result() -> ModelComparisonResult:
    """Create a sample comparison result for selection tests."""
    return ModelComparisonResult(
        leaderboard=[
            {
                "model_name": "ModelB",
                "mae": 1.0,
                "rmse": 2.0,
                "r2": 0.8,
                "mape": 5.0,
                "rank": 1,
            },
            {
                "model_name": "ModelA",
                "mae": 2.0,
                "rmse": 3.0,
                "r2": 0.7,
                "mape": 8.0,
                "rank": 2,
            },
        ],
        best_model_name="ModelB",
        ranking_metric="mae",
        comparison_time_seconds=0.01,
        number_of_models=2,
    )


def test_successful_selection(engine: BestModelSelectionEngine) -> None:
    """Selection should return the top-ranked model."""
    result = engine.select_best_model(make_comparison_result())

    assert isinstance(result, BestModelResult)
    assert result.model_name == "ModelB"
    assert result.rank == 1
    assert result.ranking_metric == "mae"


def test_correct_winner(engine: BestModelSelectionEngine) -> None:
    """The selected model should be the one ranked number one."""
    result = engine.select_best_model(make_comparison_result())

    assert result.model_name == "ModelB"
    assert result.metric_value == 1.0


def test_csv_export(engine: BestModelSelectionEngine, tmp_path: Path) -> None:
    """Selection should export the winner to CSV."""
    engine._csv_path = tmp_path / "best_model.csv"
    engine._json_path = tmp_path / "best_model.json"

    result = engine.select_best_model(make_comparison_result())
    csv_path, json_path = engine.export_selection(result)

    assert csv_path.exists()
    assert json_path.exists()
    assert "ModelB" in csv_path.read_text(encoding="utf-8")


def test_json_export(engine: BestModelSelectionEngine, tmp_path: Path) -> None:
    """Selection should export the winner to JSON."""
    engine._csv_path = tmp_path / "best_model.csv"
    engine._json_path = tmp_path / "best_model.json"

    result = engine.select_best_model(make_comparison_result())
    _, json_path = engine.export_selection(result)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "ModelB"
    assert payload["rank"] == 1


def test_empty_leaderboard_raises_value_error(engine: BestModelSelectionEngine) -> None:
    """An empty leaderboard should be rejected."""
    comparison_result = ModelComparisonResult(
        leaderboard=[],
        best_model_name="",
        ranking_metric="mae",
        comparison_time_seconds=0.0,
        number_of_models=0,
    )
    with pytest.raises(ValueError, match="leaderboard cannot be empty"):
        engine.select_best_model(comparison_result)


def test_none_comparison_result_raises_value_error(engine: BestModelSelectionEngine) -> None:
    """A None comparison result should be rejected."""
    with pytest.raises(ValueError, match="comparison_result cannot be None"):
        engine.select_best_model(None)


def test_missing_rank_1_raises_value_error(engine: BestModelSelectionEngine) -> None:
    """A leaderboard without rank 1 should be rejected."""
    comparison_result = ModelComparisonResult(
        leaderboard=[
            {
                "model_name": "ModelA",
                "mae": 2.0,
                "rmse": 3.0,
                "r2": 0.7,
                "mape": 8.0,
                "rank": 2,
            }
        ],
        best_model_name="ModelA",
        ranking_metric="mae",
        comparison_time_seconds=0.0,
        number_of_models=1,
    )
    with pytest.raises(ValueError, match="leaderboard must contain rank 1"):
        engine.select_best_model(comparison_result)


def test_returned_dataclass(engine: BestModelSelectionEngine) -> None:
    """The engine should return the expected dataclass."""
    result = engine.select_best_model(make_comparison_result())

    assert isinstance(result, BestModelResult)
    assert result.model_metadata["mae"] == 1.0
