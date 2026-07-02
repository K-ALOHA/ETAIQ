"""Model comparison engine for the ETAIQ production training pipeline."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_TRAINING_CONFIG
from .evaluation import EvaluationResult
from .logging_config import TrainingLogger


@dataclass
class ModelComparisonResult:
    """Container for the comparison of multiple evaluated models."""

    leaderboard: list[dict[str, Any]]
    best_model_name: str
    ranking_metric: str
    comparison_time_seconds: float
    number_of_models: int


class ModelComparisonEngine:
    """Compare existing evaluation results by a selected ranking metric."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.comparison")
        self._export_path = DEFAULT_TRAINING_CONFIG.models_dir / "model_leaderboard.csv"

    def compare(self, evaluation_results: list[EvaluationResult], ranking_metric: str = "mae") -> ModelComparisonResult:
        """Rank evaluated models using the supplied metric."""
        self._validate_inputs(evaluation_results, ranking_metric)

        self._logger.info(
            "Comparison started",
            ranking_metric=ranking_metric,
            number_of_models=len(evaluation_results),
        )

        start_time = time.perf_counter()
        sorted_results = sorted(
            evaluation_results,
            key=lambda result: getattr(result, ranking_metric),
            reverse=ranking_metric == "r2",
        )
        leaderboard = []
        for rank, result in enumerate(sorted_results, start=1):
            leaderboard.append(
                {
                    "model_name": result.model_name,
                    "mae": result.mae,
                    "rmse": result.rmse,
                    "r2": result.r2,
                    "mape": result.mape,
                    "rank": rank,
                }
            )

        elapsed = time.perf_counter() - start_time
        best_model_name = leaderboard[0]["model_name"]
        self._logger.info(
            "Ranking complete",
            ranking_metric=ranking_metric,
            winner=best_model_name,
            execution_time_seconds=elapsed,
        )

        return ModelComparisonResult(
            leaderboard=leaderboard,
            best_model_name=best_model_name,
            ranking_metric=ranking_metric,
            comparison_time_seconds=elapsed,
            number_of_models=len(leaderboard),
        )

    def export_leaderboard(self, result: ModelComparisonResult) -> Path:
        """Persist the comparison leaderboard to a CSV file."""
        self._export_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["Rank", "Model", "MAE", "RMSE", "R2", "MAPE"]
        file_exists = self._export_path.exists()
        with self._export_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for entry in result.leaderboard:
                writer.writerow(
                    {
                        "Rank": entry["rank"],
                        "Model": entry["model_name"],
                        "MAE": entry["mae"],
                        "RMSE": entry["rmse"],
                        "R2": entry["r2"],
                        "MAPE": entry["mape"],
                    }
                )

        return self._export_path

    def _validate_inputs(self, evaluation_results: list[EvaluationResult], ranking_metric: str) -> None:
        """Validate the comparison inputs before ranking."""
        if not evaluation_results:
            raise ValueError("evaluation_results cannot be empty")

        supported_metrics = {"mae", "rmse", "mape", "r2"}
        if ranking_metric not in supported_metrics:
            raise ValueError(f"Unsupported ranking metric: {ranking_metric}")

        model_names = [result.model_name for result in evaluation_results]
        if len(model_names) != len(set(model_names)):
            raise ValueError("Duplicate model names are not allowed")
