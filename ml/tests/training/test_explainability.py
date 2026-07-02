"""Unit tests for model explainability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml.training.explainability import ExplanationResult, ExplainabilityEngine


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


def test_validation_failures() -> None:
    """Explainability should reject invalid inputs."""
    engine = ExplainabilityEngine()

    with pytest.raises(ValueError, match="model cannot be None"):
        engine.explain_model(None, "Model", ["feature_1"])

    with pytest.raises(ValueError, match="feature list cannot be empty"):
        engine.explain_model(LinearRegression(), "Model", [])

    with pytest.raises(ValueError, match="feature mismatch"):
        engine.explain_prediction(LinearRegression(), np.array([[1.0]]), ["a", "b"])
