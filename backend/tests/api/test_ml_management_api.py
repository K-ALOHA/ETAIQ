"""Focused tests for the backend ML management API endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from app.api import explainability as explainability_module
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

    model = XGBRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    persistence_engine.save_model(model, "XGBRegressor")
    artifact_path = tmp_path / "XGBRegressor_v2.joblib"
    joblib.dump(model, artifact_path)

    registry_engine._storage_dir = tmp_path / "registry"
    registry_engine._storage_dir.mkdir(parents=True, exist_ok=True)
    registry_engine._models.clear()
    registry_engine.register_model("XGBRegressor", 2, artifact_path, {"mae": 0.0}, "Production")

    experiment_engine._experiments.clear()
    experiment_engine.log_experiment("XGBRegressor", "default", {}, {"mae": 0.0}, 0.01, 2)

    monitoring_engine._records.clear()
    monitoring_engine.record_predictions("XGBRegressor", [1.0])

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
    assert payload["models"][0]["model_name"] == "XGBRegressor"


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
    response = client.get("/api/v1/explainability/XGBRegressor")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "XGBRegressor"
    assert payload["feature_importance"]


def test_latest_explainability_uses_registered_production_model(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The latest explainability endpoint should use the registered production model and generate metadata from it."""
    monkeypatch.setattr(explainability_module, "_find_latest_artifact_dir", lambda _: None)

    model_path = tmp_path / "XGBRegressor_v2.joblib"
    model = XGBRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]), np.array([0.0, 1.0, 2.0]))
    joblib.dump(model, model_path)

    explainability_module.registry_engine._models.clear()
    explainability_module.registry_engine.register_model(
        "XGBRegressor",
        2,
        model_path,
        {"mae": 0.25},
        "Production",
        metadata={
            "feature_names": ["distance_km", "traffic_density", "avg_speed"],
            "target_column": "actual_delivery_time_min",
            "dataset": "eta_dataset",
            "framework": "xgboost",
            "evaluation_metrics": {"mae": 0.25, "rmse": 0.3, "r2": 0.9},
        },
    )

    response = client.get("/api/v1/explainability/latest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == "XGBRegressor"
    assert payload["metadata"]["model_name"] == "XGBRegressor"
    assert payload["metadata"]["feature_count"] == 3
    assert payload["metadata"]["target"] == "actual_delivery_time_min"
    assert payload["metadata"]["dataset"] == "eta_dataset"


def test_latest_explainability_rejects_linear_regression_fallback(tmp_path: Path) -> None:
    """The explainability endpoint should not silently fall back to LinearRegression when no Production XGBRegressor exists."""
    registry_engine._storage_dir = tmp_path / "registry"
    registry_engine._storage_dir.mkdir(parents=True, exist_ok=True)
    registry_engine._models.clear()

    model_path = tmp_path / "LinearRegression_v1.joblib"
    linear_model = LinearRegression()
    linear_model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    joblib.dump(linear_model, model_path)
    registry_engine.register_model("LinearRegression", 1, model_path, {"mae": 0.0}, "Production")

    with TestClient(app) as client:
        response = client.get("/api/v1/explainability/latest")

    assert response.status_code == 404
    assert "XGBRegressor" in response.json()["detail"]


def test_explainability_artifacts_are_generated_for_xgbregressor(tmp_path: Path) -> None:
    """Explainability artifact generation should write SHAP metadata for the Production XGBRegressor."""
    registry_engine._storage_dir = tmp_path / "registry"
    registry_engine._storage_dir.mkdir(parents=True, exist_ok=True)
    registry_engine._models.clear()

    model_path = tmp_path / "XGBRegressor_v2.joblib"
    model = XGBRegressor(n_estimators=3, max_depth=2, random_state=0)
    model.fit(np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]), np.array([0.0, 1.0, 2.0]))
    joblib.dump(model, model_path)

    registered = registry_engine.register_model(
        "XGBRegressor",
        2,
        model_path,
        {"mae": 0.1},
        "Production",
        metadata={"feature_names": ["distance_km", "traffic_density"], "target_column": "actual_delivery_time_min"},
    )

    artifact_context = explainability_module._ensure_explainability_artifacts(registered, tmp_path)
    artifact_dir = Path(artifact_context["output_dir"])

    metadata_payload = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    shap_payload = json.loads((artifact_dir / "shap_summary.json").read_text(encoding="utf-8"))

    assert metadata_payload["model_name"] == "XGBRegressor"
    assert shap_payload["model_name"] == "XGBRegressor"


def test_explainability_endpoint_uses_persisted_artifacts(client: TestClient, tmp_path: Path) -> None:
    """The explainability endpoint should read persisted explainability artifacts when they exist."""
    artifact_dir = tmp_path / "explainability" / "XGBRegressor" / "2"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(
        '{"model_name": "XGBRegressor", "version": 2, "explainability_available": true, "feature_importance_path": "'
        + str(artifact_dir / "feature_importance.json")
        + '", "shap_path": "'
        + str(artifact_dir / "shap_summary.json")
        + '"}',
        encoding="utf-8",
    )
    feature_importance_path = artifact_dir / "feature_importance.json"
    feature_importance_path.write_text(
        '{"model_name": "XGBRegressor", "method": "feature_importance", "feature_importance": {"feature_0": 0.75}, "ranked_features": [{"feature_name": "feature_0", "importance": 0.75}], "generated_at": "2026-01-01T00:00:00Z"}',
        encoding="utf-8",
    )

    registry_engine._models.clear()
    registry_engine.register_model(
        "XGBRegressor",
        2,
        tmp_path / "XGBRegressor_v2.joblib",
        {"mae": 0.0},
        "Production",
        metadata={
            "explainability_available": True,
            "feature_importance_path": str(feature_importance_path),
            "shap_path": str(artifact_dir / "shap_summary.json"),
        },
    )

    response = client.get("/api/v1/explainability/LinearRegression")

    assert response.status_code == 200
    payload = response.json()
    assert payload["feature_importance"] == {"feature_0": 0.75}
    assert payload["ranked_features"][0]["feature_name"] == "feature_0"
