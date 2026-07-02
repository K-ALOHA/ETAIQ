"""End-to-end training pipeline for the ETAIQ production training workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .comparison import ModelComparisonEngine, ModelComparisonResult
from .config import DEFAULT_TRAINING_CONFIG
from .evaluation import EvaluationEngine, EvaluationResult
from .logging_config import TrainingLogger
from .persistence import ModelPersistenceEngine, PersistenceResult
from .registry import ModelRegistry
from .selection import BestModelResult, BestModelSelectionEngine
from .trainer import TrainingEngine, TrainingResult


@dataclass
class TrainingPipelineResult:
    """Container for the complete training pipeline execution."""

    trained_models: list[TrainingResult]
    evaluation_results: list[EvaluationResult]
    comparison_result: ModelComparisonResult
    best_model: BestModelResult
    saved_model: PersistenceResult
    pipeline_time_seconds: float


class TrainingPipelineEngine:
    """Orchestrate training, evaluation, comparison, selection, and persistence."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.pipeline")
        self._registry = ModelRegistry()
        self._trainer = TrainingEngine(registry=self._registry, logger=logger)
        self._evaluator = EvaluationEngine(logger=logger)
        self._comparator = ModelComparisonEngine(logger=logger)
        self._selector = BestModelSelectionEngine(logger=logger)
        self._persistence = ModelPersistenceEngine(logger=logger)

    def run(self, X_train: Any, X_test: Any, y_train: Any, y_test: Any) -> TrainingPipelineResult:
        """Run the complete training workflow for all registered regression models."""
        self._validate_inputs(X_train, X_test, y_train, y_test)

        self._logger.info("Pipeline started")
        start_time = time.perf_counter()

        trained_models: list[TrainingResult] = []
        evaluation_results: list[EvaluationResult] = []

        for model_name in self._registry.list_models():
            self._logger.info("Training each model", model_name=model_name)
            training_result = self._trainer.train(model_name, X_train, y_train)
            trained_models.append(training_result)

            self._logger.info("Evaluation complete", model_name=model_name)
            evaluation_result = self._evaluator.evaluate(
                training_result.trained_model,
                model_name,
                X_test,
                y_test,
            )
            evaluation_results.append(evaluation_result)

        comparison_result = self._comparator.compare(evaluation_results, ranking_metric="mae")
        self._logger.info("Comparison complete", ranking_metric="mae")

        best_model = self._selector.select_best_model(comparison_result)
        self._logger.info("Best model selected", model_name=best_model.model_name)

        saved_model = self._persistence.save_model(
            self._find_trained_model(trained_models, best_model.model_name).trained_model,
            best_model.model_name,
            metadata={"ranking_metric": best_model.ranking_metric, "metric_value": best_model.metric_value},
        )
        self._logger.info("Model saved", model_name=best_model.model_name)

        elapsed = time.perf_counter() - start_time
        self._logger.info("Pipeline completed", pipeline_time_seconds=elapsed)

        return TrainingPipelineResult(
            trained_models=trained_models,
            evaluation_results=evaluation_results,
            comparison_result=comparison_result,
            best_model=best_model,
            saved_model=saved_model,
            pipeline_time_seconds=elapsed,
        )

    def _validate_inputs(self, X_train: Any, X_test: Any, y_train: Any, y_test: Any) -> None:
        """Validate the training and evaluation inputs before running the pipeline."""
        X_train_array = np.asarray(X_train)
        X_test_array = np.asarray(X_test)
        y_train_array = np.asarray(y_train)
        y_test_array = np.asarray(y_test)

        if X_train_array.size == 0 or X_test_array.size == 0 or y_train_array.size == 0 or y_test_array.size == 0:
            raise ValueError("Training and testing data cannot be empty")

        if len(X_train_array) != len(y_train_array):
            raise ValueError("Training features and targets must have the same length")

        if len(X_test_array) != len(y_test_array):
            raise ValueError("Testing features and targets must have the same length")

    def _find_trained_model(self, trained_models: list[TrainingResult], model_name: str) -> TrainingResult:
        """Find the training result matching the requested model name."""
        for training_result in trained_models:
            if training_result.model_name == model_name:
                return training_result
        raise ValueError(f"No trained model found for {model_name}")
