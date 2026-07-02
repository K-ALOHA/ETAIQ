"""Best model selection engine for the ETAIQ production training pipeline."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .comparison import ModelComparisonResult
from .config import DEFAULT_TRAINING_CONFIG
from .logging_config import TrainingLogger


@dataclass
class BestModelResult:
    """Container for the selected best model from a comparison result."""

    model_name: str
    ranking_metric: str
    metric_value: float
    rank: int
    selection_time_seconds: float
    model_metadata: dict[str, Any]


class BestModelSelectionEngine:
    """Select the top-ranked model from an existing comparison result."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.selection")
        self._csv_path = DEFAULT_TRAINING_CONFIG.models_dir / "best_model.csv"
        self._json_path = DEFAULT_TRAINING_CONFIG.models_dir / "best_model.json"

    def select_best_model(self, comparison_result: ModelComparisonResult | None) -> BestModelResult:
        """Select the top-ranked model from an existing comparison result."""
        self._validate_inputs(comparison_result)

        self._logger.info(
            "Selection started",
            ranking_metric=comparison_result.ranking_metric,
            number_of_models=comparison_result.number_of_models,
        )

        start_time = time.perf_counter()
        best_entry = next(entry for entry in comparison_result.leaderboard if entry["rank"] == 1)
        elapsed = time.perf_counter() - start_time

        result = BestModelResult(
            model_name=best_entry["model_name"],
            ranking_metric=comparison_result.ranking_metric,
            metric_value=float(best_entry[self._metric_key(comparison_result.ranking_metric)]),
            rank=best_entry["rank"],
            selection_time_seconds=elapsed,
            model_metadata={
                "mae": best_entry["mae"],
                "rmse": best_entry["rmse"],
                "r2": best_entry["r2"],
                "mape": best_entry["mape"],
            },
        )

        self._logger.info(
            "Selection completed",
            selected_model=result.model_name,
            metric_value=result.metric_value,
            ranking_metric=result.ranking_metric,
            execution_time_seconds=result.selection_time_seconds,
        )

        return result

    def export_selection(self, result: BestModelResult) -> tuple[Path, Path]:
        """Persist the selected best model to CSV and JSON files."""
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        with self._csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Model", "Rank", "Ranking Metric", "Metric Value"])
            writer.writeheader()
            writer.writerow(
                {
                    "Model": result.model_name,
                    "Rank": result.rank,
                    "Ranking Metric": result.ranking_metric,
                    "Metric Value": result.metric_value,
                }
            )

        payload = {
            "model_name": result.model_name,
            "ranking_metric": result.ranking_metric,
            "metric_value": result.metric_value,
            "rank": result.rank,
            "selection_time_seconds": result.selection_time_seconds,
        }
        with self._json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        return self._csv_path, self._json_path

    def _validate_inputs(self, comparison_result: ModelComparisonResult | None) -> None:
        """Validate the comparison result before selection."""
        if comparison_result is None:
            raise ValueError("comparison_result cannot be None")

        if not comparison_result.leaderboard:
            raise ValueError("leaderboard cannot be empty")

        if not any(entry["rank"] == 1 for entry in comparison_result.leaderboard):
            raise ValueError("leaderboard must contain rank 1")

    def _metric_key(self, ranking_metric: str) -> str:
        """Map ranking metric names to the leaderboard field names."""
        return ranking_metric
