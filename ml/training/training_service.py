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
from .explainability_artifacts import ExplainabilityArtifactGenerator
from .hyperparameter_search import HyperparameterSearchEngine, HyperparameterSearchResult
from .logging_config import TrainingLogger
from .model_registry import ModelRegistryEngine, RegisteredModel
from .persistence import ModelPersistenceEngine, PersistenceResult
from .pipeline import TrainingPipelineEngine
from .registry import ModelRegistry
from .selection import BestModelResult, BestModelSelectionEngine
from .trainer import TrainingEngine, TrainingResult
from ml.features.encoding import EncodingEngine
from ml.features.scaling import ScalingEngine
from ml.features.selection import FeatureSelectionEngine
import pandas as pd
from pathlib import Path


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
        artifacts_root = self._persistence_engine._models_dir.parent / "ml" / "artifacts" / "explainability"
        self._explainability_artifact_generator = ExplainabilityArtifactGenerator(artifacts_root=artifacts_root)

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
            # Ensure test data is transformed the same way as training data using the
            # fitted engines inside the TrainingEngine instance (avoid refitting).
            # Prefer using the fitted preprocessor inside the trained pipeline to prepare evaluation input
            try:
                pipeline = training_result.trained_model
                preprocessor = None
                if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
                    preprocessor = pipeline.named_steps["preprocessor"]
                if preprocessor is not None:
                    evaluation_input = preprocessor.transform(X_test)
                else:
                    # fallback to older engine-based transform
                    import pandas as _pd

                    X_train_df = X_train if isinstance(X_train, _pd.DataFrame) else _pd.DataFrame(X_train)
                    X_test_df = X_test if isinstance(X_test, _pd.DataFrame) else _pd.DataFrame(X_test)

                    enc = self._training_engine._encoding_engine
                    sca = self._training_engine._scaling_engine
                    sel = self._training_engine._selection_engine

                    encoded_train, encoded_test = enc.transform(X_train_df, X_test_df)
                    try:
                        scaled_train, scaled_test = sca.transform(encoded_train, encoded_test)
                    except Exception:
                        sca.fit(encoded_train, plan_path=None)
                        scaled_train, scaled_test = sca.transform(encoded_train, encoded_test)

                    _, selected_test = sel.select_features(scaled_train, scaled_test, _pd.Series(y_train))
                    evaluation_input = selected_test
            except Exception:
                evaluation_input = X_test

            # Prefer the exact selected features used during training if available
            if getattr(training_result, "selected_features", None):
                cols = training_result.selected_features
                try:
                    evaluation_input = evaluation_input[cols].copy()
                except Exception:
                    # fallback: keep evaluation_input as-is
                    pass

            # If we already transformed the test data using the pipeline preprocessor,
            # call the estimator directly to avoid double-preprocessing.
            if (
                isinstance(evaluation_input, (list, tuple)) is False
                and hasattr(training_result.trained_model, "named_steps")
                and "preprocessor" in training_result.trained_model.named_steps
            ):
                pipeline_obj = training_result.trained_model
                preprocessor_obj = pipeline_obj.named_steps.get("preprocessor")
                estimator_obj = pipeline_obj.named_steps.get("estimator", pipeline_obj)
                try:
                    # If evaluation_input appears already preprocessed (columns match estimator input), use estimator
                    evaluation_result = self._evaluation_engine.evaluate(
                        estimator_obj,
                        model_name,
                        evaluation_input,
                        y_test,
                    )
                except Exception:
                    # fallback to evaluating full pipeline
                    evaluation_result = self._evaluation_engine.evaluate(
                        training_result.trained_model,
                        model_name,
                        evaluation_input,
                        y_test,
                    )
            else:
                evaluation_result = self._evaluation_engine.evaluate(
                    training_result.trained_model,
                    model_name,
                    evaluation_input,
                    y_test,
                )
            evaluation_results.append(evaluation_result)

            try:
                cross_validation_result = self._cross_validation_engine.cross_validate(
                    model_name,
                    X_train,
                    y_train,
                    n_splits=min(3, len(np.asarray(X_train))),
                )
                cross_validation_results.append(cross_validation_result)
            except Exception:
                # on failure, append a placeholder CV result
                cross_validation_results.append(CrossValidationResult(
                    model_name=model_name,
                    fold_results=[],
                    mean_mae=float("nan"),
                    std_mae=float("nan"),
                    mean_rmse=float("nan"),
                    std_rmse=float("nan"),
                    mean_r2=float("nan"),
                    std_r2=float("nan"),
                    mean_mape=float("nan"),
                    std_mape=float("nan"),
                    total_training_time_seconds=0.0,
                    number_of_folds=0,
                ))

            try:
                hyperparameter_result = self._hyperparameter_engine.search(
                    model_name,
                    X_train,
                    y_train,
                    param_grid=self._registry.get_hyperparameter_grid(model_name),
                    cv=min(3, len(np.asarray(X_train))),
                )
                hyperparameter_results.append(hyperparameter_result)
            except Exception:
                hyperparameter_results.append(
                    HyperparameterSearchResult(
                        model_name=model_name,
                        best_parameters={},
                        best_score=float("nan"),
                        best_model=None,
                        all_results=[],
                        search_time_seconds=0.0,
                        number_of_configurations=0,
                    )
                )

        comparison_result = self._comparison_engine.compare(evaluation_results, ranking_metric="mae")
        best_model = self._selection_engine.select_best_model(comparison_result)
        best_training_result = self._find_trained_model(trained_models, best_model.model_name)
        best_pipeline = best_training_result.trained_model

        explainability_metadata: dict[str, Any] = {
            "ranking_metric": best_model.ranking_metric,
            "metric_value": best_model.metric_value,
            "explainability_available": False,
        }
        try:
            feature_names = self._resolve_feature_names(best_pipeline, X_train, best_training_result.selected_features)
            artifact_summary = self._explainability_artifact_generator.generate_for_model(
                self._get_model_for_explainability(best_pipeline),
                best_model.model_name,
                feature_names,
                version=self._persistence_engine._next_version(best_model.model_name),
            )
            explainability_metadata.update({
                "explainability_available": True,
                "feature_importance_path": artifact_summary["feature_importance_path"],
                "feature_importance_csv_path": artifact_summary["feature_importance_csv_path"],
                "shap_path": artifact_summary["shap_summary_path"],
                "metadata_path": artifact_summary["metadata_path"],
                "feature_names": feature_names,
                "generated_at": artifact_summary["metadata"]["generated_at"],
            })
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning(
                "Explainability artifact generation failed",
                model_name=best_model.model_name,
                error=str(exc),
            )

        saved_model = self._persistence_engine.save_model(
            best_pipeline,
            best_model.model_name,
            metadata=explainability_metadata,
        )

        # Extract all metrics from the comparison result for the best model
        best_metrics_entry = next(
            entry for entry in comparison_result.leaderboard
            if entry["model_name"] == best_model.model_name
        )
        all_metrics = {
            "mae": float(best_metrics_entry["mae"]),
            "rmse": float(best_metrics_entry["rmse"]),
            "r2": float(best_metrics_entry["r2"]),
            "mape": float(best_metrics_entry["mape"]),
        }

        experiment = self._experiment_engine.log_experiment(
            model_name=best_model.model_name,
            dataset_version="default",
            hyperparameters={"selected_model": best_model.model_name},
            metrics=all_metrics,
            training_time_seconds=sum(result.training_time_seconds for result in trained_models),
            model_version=saved_model.version,
        )

        registry_entry = self._model_registry_engine.register_model(
            model_name=best_model.model_name,
            version=saved_model.version,
            artifact_path=saved_model.model_path,
            metrics=all_metrics,
            status="Staging",
            metadata={
                **explainability_metadata,
                "artifact_path": str(saved_model.model_path),
                "saved_timestamp": saved_model.saved_timestamp,
            },
        )
        
        # If best model is LinearRegression, archive it and do NOT promote to production
        if best_model.model_name == "LinearRegression":
            self._model_registry_engine.archive_model(best_model.model_name, saved_model.version)
            self._logger.info(
                "LinearRegression is baseline model - archived, not promoted to production",
                model_name=best_model.model_name,
                version=saved_model.version,
            )
        else:
            # Otherwise promote the selected model to production
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

    def _resolve_feature_names(self, pipeline: Any, X_train: Any, selected_features: list[str] | None) -> list[str]:
        """Infer feature names from selected features, preprocessor metadata, or the raw input schema."""
        if selected_features:
            return [str(name) for name in selected_features]

        if hasattr(pipeline, "named_steps"):
            preprocessor = pipeline.named_steps.get("preprocessor")
            if preprocessor is not None:
                output_feature_names = getattr(preprocessor, "output_feature_names_", None)
                if output_feature_names:
                    return [str(name) for name in output_feature_names]
                get_feature_names_out = getattr(preprocessor, "get_feature_names_out", None)
                if callable(get_feature_names_out):
                    try:
                        names = get_feature_names_out()
                        if names is not None:
                            return [str(name) for name in names]
                    except Exception:
                        pass

        if hasattr(X_train, "columns"):
            return [str(column) for column in X_train.columns]

        array = np.asarray(X_train)
        if array.ndim == 1:
            return ["feature_0"]
        return [f"feature_{index}" for index in range(array.shape[1])]

    def _get_model_for_explainability(self, pipeline: Any) -> Any:
        """Return the estimator component for explainability generation."""
        if hasattr(pipeline, "named_steps"):
            return pipeline.named_steps.get("estimator", pipeline)
        return pipeline
