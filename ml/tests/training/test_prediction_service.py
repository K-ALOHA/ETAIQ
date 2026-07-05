"""Unit tests for the prediction service integration layer."""

from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from ml.training.prediction_service import PredictionService, PredictionServiceResult
from ml.training.model_registry import ModelRegistryEngine
from ml.training.persistence import ModelPersistenceEngine


@pytest.fixture
def service() -> PredictionService:
    """Create a prediction service for tests."""
    return PredictionService()


@pytest.fixture
def model_path(tmp_path) -> str:
    """Create a persisted regression model for prediction tests."""
    model = LinearRegression()
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    model.fit(X, y)
    path = tmp_path / "LinearRegression_v1.joblib"
    joblib.dump(model, path)
    return str(path)


def test_successful_prediction(service: PredictionService, model_path: str) -> None:
    """A valid model and input should produce a result object with all integrations."""
    result = service.predict(model_path, [[4.0]])

    assert isinstance(result, PredictionServiceResult)
    assert result.prediction is not None
    assert result.explanation is not None
    assert result.monitoring_record is not None
    assert result.prediction_time >= 0.0
    assert len(service.monitoring_engine.list_records()) == 1


def test_invalid_model_raises_value_error(service: PredictionService) -> None:
    """Missing models should be rejected before predictions are attempted."""
    with pytest.raises(ValueError, match="Model file missing"):
        service.predict("/tmp/does-not-exist.joblib", [[1.0]])


def test_invalid_input_raises_value_error(service: PredictionService, model_path: str) -> None:
    """Empty or malformed input should be rejected."""
    with pytest.raises(ValueError, match="empty"):
        service.predict(model_path, [])


def test_monitoring_integration(service: PredictionService, model_path: str) -> None:
    """Prediction results should be recorded by the monitoring engine."""
    service.predict(model_path, [[4.0]])

    records = service.monitoring_engine.list_records()
    assert len(records) == 1
    assert records[0].model_name == "LinearRegression"
    assert records[0].prediction_count == 1


def test_drift_detection(service: PredictionService, model_path: str) -> None:
    """Prediction should detect drift when a baseline has been configured."""
    service.set_drift_baseline([[0.0], [1.0], [2.0]])
    result = service.predict(model_path, [[10.0]])

    assert result.drift_result is not None
    assert any(item.drift_detected for item in result.drift_result)


def test_explainability_output(service: PredictionService, model_path: str) -> None:
    """Prediction results should include explanation metadata."""
    result = service.predict(model_path, [[4.0]])

    assert result.explanation is not None
    assert result.explanation.model_name == "LinearRegression"
    assert result.explanation.feature_importance


def test_prediction_persists_explainability_artifacts(service: PredictionService, model_path: str, tmp_path: Path) -> None:
    """A successful prediction should persist explainability artifacts for the active model."""
    service._persistence_engine._models_dir = tmp_path
    service._persistence_engine._models_dir.mkdir(parents=True, exist_ok=True)
    service._prediction_pipeline._persistence_engine._models_dir = tmp_path
    service._prediction_pipeline._persistence_engine._models_dir.mkdir(parents=True, exist_ok=True)

    service.predict(model_path, [[4.0]])

    artifact_root = tmp_path.parent / "ml" / "artifacts" / "explainability" / "LinearRegression" / "1"
    assert artifact_root.exists()
    assert (artifact_root / "feature_importance.json").exists()
    assert (artifact_root / "local_explanation.json").exists()
    assert (artifact_root / "metadata.json").exists()


def test_failure_propagation(service: PredictionService, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected upstream errors should propagate through the service."""

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(service._prediction_pipeline, "predict", _raise)

    with pytest.raises(RuntimeError, match="boom"):
        service.predict("/tmp/example.joblib", [[1.0]])


def test_registry_production_model_loaded_and_cached(tmp_path: Path) -> None:
    """PredictionService should load the registry production model and cache it."""
    persistence = ModelPersistenceEngine()
    persistence._models_dir = tmp_path
    persistence._models_dir.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistryEngine(storage_dir=tmp_path / "registry")

    model_v1 = LinearRegression()
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    model_v1.fit(X, y)
    artifact_v1 = tmp_path / "LinearRegression_v1.joblib"
    joblib.dump(model_v1, artifact_v1)

    model_v2 = LinearRegression()
    model_v2.fit(X, y)
    artifact_v2 = tmp_path / "LinearRegression_v2.joblib"
    joblib.dump(model_v2, artifact_v2)

    registry.register_model("LinearRegression", 1, artifact_v1, metrics={"mae": 0.1}, status="Production")
    registry.register_model("LinearRegression", 2, artifact_v2, metrics={"mae": 0.05}, status="Staging")

    service = PredictionService(
        registry_engine=registry,
        persistence_engine=persistence,
        model_name="LinearRegression",
    )

    # First call should load v1 from registry
    service.predict(str(artifact_v1), [[4.0]])
    assert service._cached_model_version == 1
    assert service._cached_model_path == artifact_v1

    # Promote v2 and ensure subsequent prediction reloads the new production model
    registry.set_production("LinearRegression", 2)
    service.predict(str(artifact_v1), [[4.0]])
    assert service._cached_model_version == 2
    assert service._cached_model_path == artifact_v2


def test_registry_fallback_to_provided_model_path(tmp_path: Path) -> None:
    """If no production model is available, PredictionService should use the provided model_path."""
    persistence = ModelPersistenceEngine()
    persistence._models_dir = tmp_path
    persistence._models_dir.mkdir(parents=True, exist_ok=True)

    model = LinearRegression()
    X = np.array([[0.0], [1.0]], dtype=float)
    y = np.array([0.0, 1.0], dtype=float)
    model.fit(X, y)
    fallback_path = tmp_path / "LinearRegression_v1.joblib"
    joblib.dump(model, fallback_path)

    service = PredictionService(
        registry_engine=ModelRegistryEngine(storage_dir=tmp_path / "registry"),
        persistence_engine=persistence,
        model_name="LinearRegression",
    )

    service.predict(str(fallback_path), [[2.0]])
    assert service._cached_model_path == fallback_path
    assert service._cached_model_version is None
