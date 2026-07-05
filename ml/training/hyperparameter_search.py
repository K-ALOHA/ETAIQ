"""Hyperparameter search engine for the ETAIQ production training pipeline."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from ml.features.sklearn_preprocessor import SklearnPreprocessor

from .config import DEFAULT_TRAINING_CONFIG
from .logging_config import TrainingLogger
from .registry import ModelRegistry


@dataclass
class HyperparameterSearchResult:
    """Container for a completed hyperparameter search run."""

    model_name: str
    best_parameters: dict[str, Any]
    best_score: float
    best_model: Any
    all_results: list[dict[str, Any]]
    search_time_seconds: float
    number_of_configurations: int


class HyperparameterSearchEngine:
    """Hyperparameter search engine built on top of the training registry."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.hyperparameter_search")
        self._registry = ModelRegistry()
        self._export_path = DEFAULT_TRAINING_CONFIG.models_dir / "hyperparameter_search_results.csv"

    def search(
        self,
        model_name: str,
        X: Any,
        y: Any,
        param_grid: dict[str, list[Any]],
        scoring: str = "neg_mean_absolute_error",
        cv: int = 5,
    ) -> HyperparameterSearchResult:
        """Run GridSearchCV and return the best configuration and model."""
        self._validate_inputs(model_name, X, y, param_grid, cv)

        X_array = np.asarray(X)
        y_array = np.asarray(y)

        self._logger.info(
            "Search started",
            model_name=model_name,
            number_of_parameter_combinations=len(self._parameter_product(param_grid)),
            cv_folds=cv,
        )

        start_time = time.perf_counter()
        # Wrap estimator in a pipeline with the project preprocessor so GridSearchCV
        # receives raw features and fits preprocessing per fold consistently.
        preprocessor = SklearnPreprocessor()
        estimator = self._registry.get_model(model_name)
        pipeline = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])

        # Adjust param_grid to prefix estimator parameters (pipeline step name 'estimator')
        adjusted_grid = {f"estimator__{k}": v for k, v in param_grid.items()} if param_grid else {}

        search = GridSearchCV(estimator=pipeline, param_grid=adjusted_grid, scoring=scoring, cv=cv)
        search.fit(X_array, y_array)
        elapsed = time.perf_counter() - start_time

        self._logger.info(
            "Search completed",
            model_name=model_name,
            best_parameters=search.best_params_,
            best_score=search.best_score_,
            search_time_seconds=elapsed,
        )

        all_results = [
            {
                "params": params,
                "mean_score": float(score),
                "rank": rank,
            }
            for params, score, rank in zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_score"],
                search.cv_results_["rank_test_score"],
            )
        ]

        return HyperparameterSearchResult(
            model_name=model_name,
            best_parameters=search.best_params_,
            best_score=float(search.best_score_),
            best_model=search.best_estimator_,
            all_results=all_results,
            search_time_seconds=elapsed,
            number_of_configurations=len(all_results),
        )

    def export_results(self, result: HyperparameterSearchResult) -> Path:
        """Persist hyperparameter search results to a CSV file."""
        self._export_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["model_name", "parameter_combination", "mean_score", "rank"]
        file_exists = self._export_path.exists()
        with self._export_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for index, entry in enumerate(result.all_results, start=1):
                writer.writerow(
                    {
                        "model_name": result.model_name,
                        "parameter_combination": f"config_{index}",
                        "mean_score": entry["mean_score"],
                        "rank": entry["rank"],
                    }
                )

        return self._export_path

    def _validate_inputs(self, model_name: str, X: Any, y: Any, param_grid: dict[str, list[Any]], cv: int) -> None:
        """Validate model name, data, parameter grid, and cross-validation folds."""
        if not self._registry.has_model(model_name):
            raise ValueError(f"Unknown model name: {model_name}")

        X_array = np.asarray(X)
        y_array = np.asarray(y)

        if X_array.size == 0 or y_array.size == 0:
            raise ValueError("Training data cannot be empty")

        if len(X_array) != len(y_array):
            raise ValueError("Training features and targets must have the same length")

        if not param_grid:
            raise ValueError("Parameter grid cannot be empty")

        if cv < 2:
            raise ValueError("cv must be at least 2")

    def _parameter_product(self, param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """Compute the Cartesian product of parameter values."""
        if not param_grid:
            return []

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        product: list[dict[str, Any]] = []
        for combination in np.array(np.meshgrid(*values, indexing="ij")).T.reshape(-1, len(keys)):
            product.append({key: combination[index] for index, key in enumerate(keys)})
        return product
