"""Model registry for the ETAIQ training pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
try:
    from xgboost import XGBRegressor
except Exception as exc:  # cover ImportError and runtime load errors (libomp, etc.)
    raise ImportError(
        "xgboost is required for XGBRegressor support. Ensure xgboost and its native dependencies (e.g. libomp) are installed."
    ) from exc

from .logging_config import TrainingLogger
from .models import ModelDefinition


class ModelRegistry:
    """Registry of available production regression models."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.registry")
        self._models: dict[str, tuple[type[Any], ModelDefinition]] = {}
        self._register_default_models()

    def _register_default_models(self) -> None:
        """Register the built-in regression models and their metadata."""
        definitions = [
            (
                "LinearRegression",
                LinearRegression,
                ModelDefinition(
                    model_name="LinearRegression",
                    task="regression",
                    needs_scaling=False,
                    supports_feature_importance=False,
                    default_parameters={},
                ),
            ),
            (
                "RandomForestRegressor",
                RandomForestRegressor,
                ModelDefinition(
                    model_name="RandomForestRegressor",
                    task="regression",
                    needs_scaling=False,
                    supports_feature_importance=True,
                    default_parameters={"n_estimators": 100, "random_state": 42},
                ),
            ),
            (
                "GradientBoostingRegressor",
                GradientBoostingRegressor,
                ModelDefinition(
                    model_name="GradientBoostingRegressor",
                    task="regression",
                    needs_scaling=False,
                    supports_feature_importance=True,
                    default_parameters={"random_state": 42},
                ),
            ),
            (
                "XGBRegressor",
                XGBRegressor,
                ModelDefinition(
                    model_name="XGBRegressor",
                    task="regression",
                    needs_scaling=False,
                    supports_feature_importance=True,
                    default_parameters={
                        "objective": "reg:squarederror",
                        "random_state": 42,
                        "n_estimators": 100,
                    },
                ),
            ),
        ]

        for name, model_cls, metadata in definitions:
            self._models[name] = (model_cls, metadata)
            self._logger.info(
                "Registered training model",
                model_name=name,
                task=metadata.task,
                needs_scaling=metadata.needs_scaling,
            )

    def list_models(self) -> list[str]:
        """Return the registered model names in registration order."""
        return list(self._models.keys())

    def has_model(self, name: str) -> bool:
        """Return whether a model with the given name is registered."""
        return name in self._models

    def get_model(self, name: str) -> Any:
        """Return a fresh sklearn model instance for the requested name."""
        if not self.has_model(name):
            raise ValueError(f"Unknown model name: {name}")

        model_cls, metadata = self._models[name]
        # If the model class is unavailable (optional dependency missing), raise a clear error
        if model_cls is None:
            raise RuntimeError(f"Model class for {name} is unavailable. Is the optional dependency installed?")

        # Instantiate with registered default parameters when provided
        try:
            return model_cls(**(metadata.default_parameters or {}))
        except TypeError:
            # Fallback to no-arg construction if parameters aren't accepted
            return model_cls()

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Return a serializable metadata dictionary for the requested model."""
        if not self.has_model(name):
            raise ValueError(f"Unknown model name: {name}")

        _, metadata = self._models[name]
        return {
            "model_name": metadata.model_name,
            "task": metadata.task,
            "needs_scaling": metadata.needs_scaling,
            "supports_feature_importance": metadata.supports_feature_importance,
            "default_parameters": metadata.default_parameters,
        }
