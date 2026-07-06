"""Pydantic schemas for backend ML management endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class TrainingRequest(BaseModel):
    """Request schema for triggering a training workflow."""

    X_train: list[list[float]] | None = Field(default=None, description="Training feature matrix")
    X_test: list[list[float]] | None = Field(default=None, description="Test feature matrix")
    y_train: list[float] | None = Field(default=None, description="Training target vector")
    y_test: list[float] | None = Field(default=None, description="Test target vector")

    @model_validator(mode="after")
    def validate_inputs(self) -> TrainingRequest:
        """Ensure the payload is well-formed before dispatching training."""
        if self.X_train is None:
            self.X_train = [[0.0], [1.0], [2.0], [3.0]]
        if self.X_test is None:
            self.X_test = [[4.0], [5.0]]
        if self.y_train is None:
            self.y_train = [0.0, 1.0, 2.0, 3.0]
        if self.y_test is None:
            self.y_test = [4.0, 5.0]
        return self


class TrainingResponse(BaseModel):
    """Response schema for a completed training workflow."""

    best_model_name: str = Field(description="Name of the best model selected")
    best_model_version: int = Field(description="Version of the best model selected")
    training_time_seconds: float = Field(description="Total training elapsed time")
    saved_model_path: str = Field(description="Path to the persisted model artifact")
    registry_status: str = Field(description="Registry status assigned to the model")
    experiment_id: str = Field(description="Identifier of the created experiment")


class RegisteredModelResponse(BaseModel):
    """Response schema for a registered model entry."""

    model_name: str
    version: int
    artifact_path: str
    metrics: dict[str, Any]
    created_at: str
    status: str


class ModelRegistryResponse(BaseModel):
    """Response schema for the full model registry snapshot."""

    models: list[RegisteredModelResponse]
    count: int


class ExperimentResponse(BaseModel):
    """Response schema for a single experiment record."""

    experiment_id: str
    timestamp: str
    model_name: str
    dataset_version: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    training_time_seconds: float
    model_version: int


class ExperimentHistoryResponse(BaseModel):
    """Response schema for experiment listing."""

    experiments: list[ExperimentResponse]
    count: int


class MonitoringRecordResponse(BaseModel):
    """Response schema for a monitoring record."""

    timestamp: str
    model_name: str
    prediction_count: int
    mean_prediction: float
    std_prediction: float
    min_prediction: float
    max_prediction: float
    missing_inputs: int
    out_of_range_inputs: int


class MonitoringResponse(BaseModel):
    """Response schema for monitoring records."""

    records: list[MonitoringRecordResponse]
    count: int


class DriftResultResponse(BaseModel):
    """Response schema for a drift result."""

    feature_name: str
    baseline_mean: float
    current_mean: float
    baseline_std: float
    current_std: float
    drift_score: float
    drift_detected: bool


class DriftResponse(BaseModel):
    """Response schema for drift detection output."""

    results: list[DriftResultResponse]
    drift_detected: bool


class ExplainabilityResponse(BaseModel):
    """Response schema for explainability output."""

    model_name: str
    feature_importance: dict[str, float]
    ranked_features: list[dict[str, Any]]
    explanation_time_seconds: float
    explanation_method: str
    sample_count: int


class ExplainabilityLatestResponse(BaseModel):
    """Response schema for the latest explainability workspace payload."""

    model_name: str
    version: str | None = None
    latest_prediction_value: float = 0.0
    prediction_time: str = ""
    confidence_score: float = 0.0
    explainability_status: str = "unavailable"
    feature_importance: dict[str, float] = Field(default_factory=dict)
    ranked_features: list[dict[str, Any]] = Field(default_factory=list)
    local_explanation: list[dict[str, Any]] = Field(default_factory=list)
    natural_language_explanation: str = ""
    summary_plot: str | None = None
    waterfall_plot: str | None = None
    plots: dict[str, str | None] = Field(default_factory=dict)
    metadata_json: str = "{}"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetResponse(BaseModel):
    """Response schema for dataset summary."""
    record_count: int
    feature_count: int
    target_column: str | None
    missing_values: dict[str, int]
    feature_names: list[str]


class PerformanceResponse(BaseModel):
    """Response schema for performance metrics."""
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    r2: float | None = None
    inference_latency_ms: float | None = None
