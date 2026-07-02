"""Pydantic schemas for prediction API endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request schema for a single prediction."""

    features: dict[str, Any] | None = Field(
        default=None,
        description="Feature values for a single prediction row.",
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, value: dict[str, Any] | None) -> dict[str, Any]:
        """Ensure the feature payload contains a non-empty object of scalar values."""
        if value is None:
            raise ValueError("features is required")
        if not isinstance(value, dict):
            raise TypeError("features must be an object")
        if not value:
            raise ValueError("features cannot be empty")

        for key, feature_value in value.items():
            if not isinstance(key, str):
                raise TypeError("feature names must be strings")
            if feature_value is None:
                raise ValueError("feature values cannot be null")
            if isinstance(feature_value, (dict, list)):
                raise TypeError("feature values must be scalar values")

        return value


class PredictionResponse(BaseModel):
    """Response schema for a completed prediction."""

    prediction: float = Field(description="Predicted value for the supplied features")
    model_name: str = Field(description="Name of the model used for prediction")
    model_version: int = Field(description="Version of the model used for prediction")
    processing_time_ms: float = Field(description="Processing duration in milliseconds")


class ModelInfoResponse(BaseModel):
    """Response schema for model metadata."""

    current_model: str = Field(description="Name of the currently active model")
    version: int = Field(description="Version of the currently active model")
    created_at: str = Field(description="Timestamp of the active model artifact")
    available_models: list[str] = Field(description="List of available model artifacts")
    models: list[dict[str, Any]] = Field(default_factory=list, description="Registry entries for available models")
    count: int = Field(default=0, description="Number of registered model entries")


class HealthResponse(BaseModel):
    """Response schema for service health."""

    status: str = Field(default="healthy", description="Current health status")
    model_loaded: bool = Field(description="Whether a persisted model is currently available")
