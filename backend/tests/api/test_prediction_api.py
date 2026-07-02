"""Unit tests for the prediction FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from app.api import prediction as prediction_module
from app.main import app
from ml.training.persistence import ModelPersistenceEngine


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a FastAPI test client backed by a temporary persisted model."""
    monkeypatch.setattr(prediction_module.settings, "model_path", str(tmp_path))

    persistence_engine = ModelPersistenceEngine()
    persistence_engine._models_dir = tmp_path

    model = LinearRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    persistence_engine.save_model(model, "LinearRegression")

    with TestClient(app) as test_client:
        yield test_client


def test_successful_prediction(client: TestClient) -> None:
    """The prediction endpoint should return a prediction for valid payloads."""
    response = client.post("/api/v1/predict", json={"features": {"feature": 3.0}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "LinearRegression"
    assert payload["model_version"] >= 1
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
    assert payload["current_model"] == "LinearRegression_v1"
    assert payload["version"] == 1
    assert payload["available_models"] == ["LinearRegression_v1"]


def test_model_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The prediction endpoint should return 404 when no model artifacts are available."""
    monkeypatch.setattr(prediction_module.settings, "model_path", str(tmp_path))

    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"features": {"feature": 3.0}})

    assert response.status_code == 404
