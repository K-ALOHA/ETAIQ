"""Focused tests for the backend ML management API endpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from app.api import prediction as prediction_module
from app.api.experiments import experiment_engine
from app.api.models import registry_engine
from app.api.monitoring import monitoring_engine
from app.main import app
from ml.training.persistence import ModelPersistenceEngine


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a FastAPI test client backed by persisted training artifacts."""
    monkeypatch.setattr(prediction_module.settings, "model_path", str(tmp_path))

    persistence_engine = ModelPersistenceEngine()
    persistence_engine._models_dir = tmp_path

    model = LinearRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    persistence_engine.save_model(model, "LinearRegression")

    registry_engine._storage_dir = tmp_path / "registry"
    registry_engine._storage_dir.mkdir(parents=True, exist_ok=True)
    registry_engine._models.clear()
    registry_engine.register_model("LinearRegression", 1, tmp_path / "LinearRegression_v1.joblib", {"mae": 0.0}, "Production")

    experiment_engine._experiments.clear()
    experiment_engine.log_experiment("LinearRegression", "default", {}, {"mae": 0.0}, 0.01, 1)

    monitoring_engine._records.clear()
    monitoring_engine.record_predictions("LinearRegression", [1.0])

    with TestClient(app) as test_client:
        yield test_client


def test_train_endpoint(client: TestClient) -> None:
    """The training endpoint should accept a training request and return metadata."""
    response = client.post(
        "/api/v1/train",
        json={"X_train": [[0.0], [1.0], [2.0], [3.0]], "X_test": [[4.0]], "y_train": [0.0, 1.0, 2.0, 3.0], "y_test": [4.0]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["best_model_name"]
    assert payload["best_model_version"] >= 1
    assert payload["registry_status"]


def test_models_endpoint(client: TestClient) -> None:
    """The models endpoint should return registry entries."""
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1
    assert payload["models"][0]["model_name"] == "LinearRegression"


def test_experiments_endpoint(client: TestClient) -> None:
    """The experiments endpoint should return experiment history."""
    response = client.get("/api/v1/experiments")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1


def test_monitoring_endpoint(client: TestClient) -> None:
    """The monitoring endpoint should return monitoring records."""
    response = client.get("/api/v1/monitoring")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1


def test_drift_endpoint(client: TestClient) -> None:
    """The drift endpoint should return a default response when no baseline exists."""
    response = client.get("/api/v1/drift")

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == []


def test_explainability_endpoint(client: TestClient) -> None:
    """The explainability endpoint should return explainability metadata for a model."""
    response = client.get("/api/v1/explainability/LinearRegression")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "LinearRegression"
    assert payload["feature_importance"]
