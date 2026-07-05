"""Unit tests for model explainability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml.training.explainability import ExplanationResult, ExplainabilityEngine
from ml.training.explainability_artifacts import (
    ExplainabilityArtifactGenerator,
    ExplainabilityArtifactInconsistencyError
)


@dataclass
class TestRegisteredModel:
    model_name: str
    version: int
    artifact_path: str
    metrics: dict
    created_at: str
    status: str
    metadata: dict


def test_random_forest_feature_importance() -> None:
    """RandomForest models should expose feature_importances_ values."""
    engine = ExplainabilityEngine()
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0.0, 1.0, 2.0, 3.0]))

    result = engine.explain_model(model, "RandomForest", ["feature_1"])

    assert isinstance(result, ExplanationResult)
    assert result.model_name == "RandomForest"
    assert result.explanation_method == "feature_importance"
    assert result.ranked_features[0]["feature_name"] == "feature_1"


def test_linear_regression_coefficients() -> None:
    """LinearRegression models should use absolute coefficient values."""
    engine = ExplainabilityEngine()
    model = LinearRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 2.0, 4.0]))

    result = engine.explain_model(model, "LinearRegression", ["feature_1"])

    assert result.explanation_method == "coefficients"
    assert result.ranked_features[0]["importance"] >= 0.0


def test_fallback_explanation() -> None:
    """Fallback explanation should return equal weights when no supported attribution exists."""
    engine = ExplainabilityEngine()

    class DummyModel:
        pass

    result = engine.explain_model(DummyModel(), "Dummy", ["a", "b"])

    assert result.explanation_method == "fallback"
    assert result.ranked_features[0]["importance"] == 1.0 / 2.0


def test_prediction_explanation() -> None:
    """Local explanation should include feature values and contribution scores."""
    engine = ExplainabilityEngine()
    model = LinearRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 2.0, 4.0]))

    explanation = engine.explain_prediction(model, np.array([[2.0]]), ["feature_1"])

    assert explanation[0]["feature_name"] == "feature_1"
    assert explanation[0]["value"] == 2.0
    assert explanation[0]["importance"] >= 0.0
    assert explanation[0]["contribution_score"] >= 0.0


def test_exports(tmp_path: Path) -> None:
    """CSV and JSON export helpers should write files."""
    engine = ExplainabilityEngine()
    model = LinearRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 2.0, 4.0]))
    result = engine.explain_model(model, "LinearRegression", ["feature_1"])

    csv_path = engine.export_csv(result, tmp_path / "explanation.csv")
    json_path = engine.export_json(result, tmp_path / "explanation.json")

    assert csv_path.exists()
    assert json_path.exists()


def test_artifact_generation_writes_expected_files(tmp_path: Path) -> None:
    """Explainability artifacts should be persisted under the model version directory."""
    engine = ExplainabilityEngine()
    model = LinearRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 2.0, 4.0]))
    result = engine.explain_model(model, "LinearRegression", ["feature_1"], input_data=np.array([[2.0]]))

    generator = ExplainabilityArtifactGenerator(artifacts_root=tmp_path)
    test_reg_model = TestRegisteredModel(
        model_name="LinearRegression",
        version=1,
        artifact_path=str(tmp_path / "model.joblib"),
        metrics={"mae": 0.1},
        created_at=datetime.now(timezone.utc).isoformat(),
        status="Staging",
        metadata={
            "feature_count": 1,
            "target_column": "target",
            "feature_names": ["feature_1"]
        }
    )
    artifacts = generator.generate_for_model(model, test_reg_model, explanation=result)

    assert Path(artifacts["feature_importance_path"]).exists()
    assert Path(artifacts["local_explanation_path"]).exists()
    assert Path(artifacts["summary_plot_path"]).exists()
    assert Path(artifacts["waterfall_plot_path"]).exists()
    assert Path(artifacts["metadata_path"]).exists()


def test_metadata_matches_registry_entry(tmp_path: Path) -> None:
    """Generated metadata should exactly match the registry entry's metadata."""
    engine = ExplainabilityEngine()
    model = LinearRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 2.0, 4.0]))
    result = engine.explain_model(model, "LinearRegression", ["feature_1"])

    generator = ExplainabilityArtifactGenerator(artifacts_root=tmp_path)
    test_reg_model = TestRegisteredModel(
        model_name="TestModel",
        version=42,
        artifact_path=str(tmp_path / "test_model.joblib"),
        metrics={"mae": 0.5, "rmse": 0.6, "r2": 0.7},
        created_at="2024-01-01T00:00:00Z",
        status="Production",
        metadata={
            "feature_count": 2,
            "target_column": "delivery_time",
            "feature_names": ["feature_a", "feature_b"]
        }
    )
    artifacts = generator.generate_for_model(model, test_reg_model, explanation=result)
    
    with open(artifacts["metadata_path"], "r") as f:
        metadata = json.load(f)
        
    assert metadata["model_name"] == test_reg_model.model_name
    assert metadata["version"] == test_reg_model.version
    assert metadata["feature_count"] == test_reg_model.metadata["feature_count"]
    assert metadata["target"] == test_reg_model.metadata["target_column"]
    assert metadata["feature_names"] == test_reg_model.metadata["feature_names"]
    assert metadata["metrics"] == test_reg_model.metrics


def test_metadata_validation_fails_on_mismatch(tmp_path: Path) -> None:
    """Validation should raise ExplainabilityArtifactInconsistencyError when metadata doesn't match registry."""
    engine = ExplainabilityEngine()
    model = LinearRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 2.0, 4.0]))
    result = engine.explain_model(model, "LinearRegression", ["feature_1"])

    generator = ExplainabilityArtifactGenerator(artifacts_root=tmp_path)
    # Create registry model with feature count 2 but feature_names of length 1 → mismatch!
    test_reg_model = TestRegisteredModel(
        model_name="TestModel",
        version=99,
        artifact_path=str(tmp_path / "model.joblib"),
        metrics={"mae": 0.5},
        created_at=datetime.now(timezone.utc).isoformat(),
        status="Production",
        metadata={
            "feature_count": 2,  # MISMATCH!
            "target_column": "y",
            "feature_names": ["feature_1"]
        }
    )

    with pytest.raises(ExplainabilityArtifactInconsistencyError):
        generator.generate_for_model(model, test_reg_model, explanation=result)


def test_missing_shap_fallback_is_graceful() -> None:
    """When SHAP is unavailable, the engine should still provide feature-importance output."""
    engine = ExplainabilityEngine()
    model = LinearRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 2.0, 4.0]))

    result = engine.explain_model(model, "LinearRegression", ["feature_1"])

    assert result.explanation_method in {"feature_importance", "coefficients", "fallback"}
    assert result.feature_importance["feature_1"] >= 0.0


def test_confidence_and_uncertainty_are_calculated() -> None:
    """Explainability results should expose confidence and uncertainty metrics."""
    engine = ExplainabilityEngine()
    model = LinearRegression()
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 2.0, 4.0]))

    result = engine.explain_model(model, "LinearRegression", ["feature_1"])

    assert 0.0 <= result.confidence_score <= 1.0
    assert result.uncertainty_estimate == 1.0 - result.confidence_score


def test_validation_failures() -> None:
    """Explainability should reject invalid inputs."""
    engine = ExplainabilityEngine()

    with pytest.raises(ValueError, match="model cannot be None"):
        engine.explain_model(None, "Model", ["feature_1"])

    with pytest.raises(ValueError, match="feature list cannot be empty"):
        engine.explain_model(LinearRegression(), "Model", [])

    with pytest.raises(ValueError, match="feature mismatch"):
        engine.explain_prediction(LinearRegression(), np.array([[1.0]]), ["a", "b"])
