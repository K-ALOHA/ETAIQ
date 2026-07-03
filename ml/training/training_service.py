"""High-level orchestration service for ETAIQ training workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .comparison import ModelComparisonEngine, ModelComparisonResult
from .cross_validation import CrossValidationEngine, CrossValidationResult
from .evaluation import EvaluationEngine, EvaluationResult
from .experiment_tracking import ExperimentRecord, ExperimentTrackingEngine
from .hyperparameter_search import HyperparameterSearchEngine, HyperparameterSearchResult
from .logging_config import TrainingLogger
from .model_registry import ModelRegistryEngine, RegisteredModel
from .persistence import ModelPersistenceEngine, PersistenceResult
from .pipeline import TrainingPipelineEngine
from .registry import ModelRegistry
from .selection import BestModelResult, BestModelSelectionEngine
from .trainer import TrainingEngine, TrainingResult


@dataclass
class TrainingServiceResult:
    """Container for the complete training-service workflow result."""

    trained_models: list[TrainingResult]
    evaluation_results: list[EvaluationResult]
    cross_validation_results: list[CrossValidationResult]
    hyperparameter_results: list[HyperparameterSearchResult]
    comparison_result: ModelComparisonResult
    best_model: BestModelResult
    saved_model: PersistenceResult
    experiment: ExperimentRecord
    registry_entry: RegisteredModel
    training_time: float


class TrainingService:
    """Coordinate training, evaluation, tuning, persistence, registry, and experiment tracking."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.service")
        self._registry = ModelRegistry()
        self._training_engine = TrainingEngine(registry=self._registry, logger=logger)
        self._evaluation_engine = EvaluationEngine(logger=logger)
        self._cross_validation_engine = CrossValidationEngine(logger=logger)
        self._hyperparameter_engine = HyperparameterSearchEngine(logger=logger)
        self._comparison_engine = ModelComparisonEngine(logger=logger)
        self._selection_engine = BestModelSelectionEngine(logger=logger)
        self._persistence_engine = ModelPersistenceEngine(logger=logger)
        self._experiment_engine = ExperimentTrackingEngine(logger=logger)
        self._model_registry_engine = ModelRegistryEngine(logger=logger)

    def train(self, X_train: Any, X_test: Any, y_train: Any, y_test: Any) -> TrainingServiceResult:
        """Run the complete production training workflow for all registered models."""
        self._validate_inputs(X_train, X_test, y_train, y_test)
        self._logger.info("Training service started")

        start_time = time.perf_counter()

        trained_models: list[TrainingResult] = []
        evaluation_results: list[EvaluationResult] = []
        cross_validation_results: list[CrossValidationResult] = []
        hyperparameter_results: list[HyperparameterSearchResult] = []

        for model_name in self._registry.list_models():
            training_result = self._training_engine.train(model_name, X_train, y_train)
            trained_models.append(training_result)

            evaluation_result = self._evaluation_engine.evaluate(
                training_result.trained_model,
                model_name,
                X_test,
                y_test,
            )
            evaluation_results.append(evaluation_result)

            cross_validation_result = self._cross_validation_engine.cross_validate(
                model_name,
                X_train,
                y_train,
                n_splits=min(3, len(np.asarray(X_train))),
            )
            cross_validation_results.append(cross_validation_result)

            hyperparameter_result = self._hyperparameter_engine.search(
                model_name,
                X_train,
                y_train,
                param_grid=self._registry.get_hyperparameter_grid(model_name),
                cv=min(3, len(np.asarray(X_train))),
            )
            hyperparameter_results.append(hyperparameter_result)

        comparison_result = self._comparison_engine.compare(evaluation_results, ranking_metric="mae")
        best_model = self._selection_engine.select_best_model(comparison_result)
        saved_model = self._persistence_engine.save_model(
            self._find_trained_model(trained_models, best_model.model_name).trained_model,
            best_model.model_name,
            metadata={"ranking_metric": best_model.ranking_metric, "metric_value": best_model.metric_value},
        )

        experiment = self._experiment_engine.log_experiment(
            model_name=best_model.model_name,
            dataset_version="default",
            hyperparameters={"selected_model": best_model.model_name},
            metrics={"mae": float(next(entry["mae"] for entry in comparison_result.leaderboard if entry["model_name"] == best_model.model_name))},
            training_time_seconds=sum(result.training_time_seconds for result in trained_models),
            model_version=saved_model.version,
        )

        registry_entry = self._model_registry_engine.register_model(
            model_name=best_model.model_name,
            version=saved_model.version,
            artifact_path=saved_model.model_path,
            metrics={"mae": float(next(entry["mae"] for entry in comparison_result.leaderboard if entry["model_name"] == best_model.model_name))},
            status="Staging",
        )
        self._model_registry_engine.set_production(best_model.model_name, saved_model.version)

        elapsed = time.perf_counter() - start_time
        self._logger.info("Training service completed", training_time_seconds=elapsed)

        return TrainingServiceResult(
            trained_models=trained_models,
            evaluation_results=evaluation_results,
            cross_validation_results=cross_validation_results,
            hyperparameter_results=hyperparameter_results,
            comparison_result=comparison_result,
            best_model=best_model,
            saved_model=saved_model,
            experiment=experiment,
            registry_entry=self._model_registry_engine.get_model(best_model.model_name, saved_model.version),
            training_time=elapsed,
        )

    def _validate_inputs(self, X_train: Any, X_test: Any, y_train: Any, y_test: Any) -> None:
        """Validate the training and evaluation inputs before running the workflow."""
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
        """Locate the training result for a given model name."""
        for training_result in trained_models:
            if training_result.model_name == model_name:
                return training_result
        raise ValueError(f"No trained model found for {model_name}")
