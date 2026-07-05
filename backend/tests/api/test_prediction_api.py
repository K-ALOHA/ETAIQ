"""Unit tests for the prediction FastAPI endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from app.api import prediction as prediction_module
from app.api.models import registry_engine
from app.main import app
from ml.training.persistence import ModelPersistenceEngine


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a FastAPI test client backed by a temporary persisted model."""
    monkeypatch.setattr(prediction_module.settings, "model_path", str(tmp_path))

    persistence_engine = ModelPersistenceEngine()
    persistence_engine._models_dir = tmp_path

    registry_engine._storage_dir = tmp_path / "registry"
    registry_engine._storage_dir.mkdir(parents=True, exist_ok=True)
    registry_engine._models.clear()
    artifact_path = tmp_path / "XGBRegressor_v2.joblib"
    model = XGBRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    joblib.dump(model, artifact_path)
    persistence_engine.save_model(model, "XGBRegressor")

    registry_engine.register_model(
        "XGBRegressor",
        2,
        artifact_path,
        {"mae": 0.0},
        "Production",
        metadata={"dataset_size": 264777, "training_samples": 211821, "testing_samples": 52956},
    )

    registry_engine._models.clear()
    registry_engine.register_model(
        "XGBRegressor",
        2,
        artifact_path,
        {"mae": 0.0},
        "Production",
        metadata={"dataset_size": 264777, "training_samples": 211821, "testing_samples": 52956},
    )

    with TestClient(app) as test_client:
        yield test_client


def test_successful_prediction(client: TestClient) -> None:
    """The prediction endpoint should return a prediction for valid payloads."""
    response = client.post("/api/v1/predict", json={"features": {"feature": 3.0}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "XGBRegressor"
    assert payload["model_version"] == 2
    assert isinstance(payload["prediction"], (int, float))
    assert payload["processing_time_ms"] >= 0.0


def test_invalid_payload(client: TestClient) -> None:
    """The prediction endpoint should reject non-object feature payloads."""
    response = client.post("/api/v1/predict", json={"features": []})

    assert response.status_code == 400


def test_missing_features(client: TestClient) -> None:
    """The prediction endpoint should reject missing feature payloads."""
    response = client.post("/api/v1/predict", json={})

    assert response.status_code == 400


def test_health_endpoint(client: TestClient) -> None:
    """The health endpoint should report service health and model availability."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True


def test_model_endpoint(client: TestClient) -> None:
    """The model endpoint should list persisted model metadata."""
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_model"] == "XGBRegressor_v2"
    assert payload["version"] == 2
    assert payload["available_models"] == ["XGBRegressor_v2"]
    assert payload["models"][0]["dataset_size"] == 264777
    assert payload["models"][0]["training_samples"] == 211821
    assert payload["models"][0]["testing_samples"] == 52956


def test_model_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The prediction endpoint should return 404 when no model artifacts are available."""
    monkeypatch.setattr(prediction_module.settings, "model_path", str(tmp_path))

    registry_engine._storage_dir = tmp_path / "registry"
    registry_engine._storage_dir.mkdir(parents=True, exist_ok=True)
    registry_engine._models.clear()

    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"features": {"feature": 3.0}})

    assert response.status_code == 503
    assert response.json()["detail"] == "No production model is registered."


def test_prediction_prefers_latest_production_xgbregressor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The prediction endpoint should prefer the latest Production XGBRegressor over legacy LinearRegression."""
    monkeypatch.setattr(prediction_module.settings, "model_path", str(tmp_path))

    registry_engine._storage_dir = tmp_path / "registry"
    registry_engine._storage_dir.mkdir(parents=True, exist_ok=True)
    registry_engine._models.clear()

    xgb_model = XGBRegressor(n_estimators=3, max_depth=2, random_state=0)
    xgb_model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    xgb_path = tmp_path / "XGBRegressor_v2.joblib"
    joblib.dump(xgb_model, xgb_path)

    legacy_path = tmp_path / "LinearRegression_v1.joblib"
    legacy_model = LinearRegression()
    legacy_model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    joblib.dump(legacy_model, legacy_path)

    registry_engine.register_model("LinearRegression", 1, legacy_path, {"mae": 0.0}, "Production")
    registry_engine.register_model("XGBRegressor", 2, xgb_path, {"mae": 0.1}, "Production")

    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"features": {"feature": 3.0}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "XGBRegressor"
    assert payload["model_version"] == 2


def test_models_endpoint_reports_latest_production_xgbregressor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The models endpoint should report the latest Production XGBRegressor as the active model."""
    monkeypatch.setattr(prediction_module.settings, "model_path", str(tmp_path))

    registry_engine._storage_dir = tmp_path / "registry"
    registry_engine._storage_dir.mkdir(parents=True, exist_ok=True)
    registry_engine._models.clear()

    legacy_path = tmp_path / "LinearRegression_v1.joblib"
    legacy_model = LinearRegression()
    legacy_model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    joblib.dump(legacy_model, legacy_path)
    registry_engine.register_model("LinearRegression", 1, legacy_path, {"mae": 0.0}, "Production")

    xgb_path = tmp_path / "XGBRegressor_v2.joblib"
    xgb_model = XGBRegressor(n_estimators=3, max_depth=2, random_state=0)
    xgb_model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    joblib.dump(xgb_model, xgb_path)
    registry_engine.register_model("XGBRegressor", 2, xgb_path, {"mae": 0.1}, "Production")

    with TestClient(app) as client:
        response = client.get("/api/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_model"] == "XGBRegressor_v2"
    assert payload["version"] == 2


def test_prediction_requires_existing_production_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The prediction endpoint should reject a production registry entry whose artifact is missing."""
    monkeypatch.setattr(prediction_module.settings, "model_path", str(tmp_path))

    registry_engine._storage_dir = tmp_path / "registry"
    registry_engine._storage_dir.mkdir(parents=True, exist_ok=True)
    registry_engine._models.clear()

    artifact_path = tmp_path / "XGBRegressor_v2.joblib"
    artifact_path.touch()
    registry_engine.register_model("XGBRegressor", 2, artifact_path, {"mae": 0.0}, "Production")
    artifact_path.unlink()

    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"features": {"feature": 3.0}})

    assert response.status_code == 500
    assert response.json()["detail"] == f"Production artifact is missing: {artifact_path}"
