"""Integration tests for the complete ETAIQ ML lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.features.config import FeatureEngineeringConfig
from ml.features.sklearn_preprocessor import SklearnPreprocessor
from ml.training.config import TrainingConfig
from ml.training.model_registry import ModelRegistryEngine
from ml.training.persistence import ModelPersistenceEngine
from ml.training.prediction_pipeline import PredictionPipelineEngine
from ml.training.training_service import TrainingService


@pytest.fixture
def isolated_ml_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect model artifacts, registry, and preprocess artifacts into a temporary path."""
    import ml.training.persistence as persistence_mod
    import ml.training.model_registry as registry_mod

    new_config = TrainingConfig(
        project_root=tmp_path,
        models_dir=tmp_path / "models",
        model_registry_path=tmp_path / "model_registry.json",
    )
    monkeypatch.setattr(persistence_mod, "DEFAULT_TRAINING_CONFIG", new_config)

    original_registry_init = registry_mod.ModelRegistryEngine.__init__

    def patched_registry_init(self, logger=None, storage_dir=None):
        return original_registry_init(self, logger=logger, storage_dir=tmp_path / "registry")

    monkeypatch.setattr(registry_mod.ModelRegistryEngine, "__init__", patched_registry_init)

    original_preprocessor_init = SklearnPreprocessor.__init__

    def patched_preprocessor_init(self, *args: Any, **kwargs: Any) -> None:
        original_preprocessor_init(self, *args, **kwargs)
        self.encoding.config = FeatureEngineeringConfig(project_root=tmp_path)
        self.scaling.config = FeatureEngineeringConfig(project_root=tmp_path)
        self.selection.config = FeatureEngineeringConfig(project_root=tmp_path)
        self.metadata_path = tmp_path / "artifacts" / "preprocessing" / "preprocessor_metadata.json"

    monkeypatch.setattr(SklearnPreprocessor, "__init__", patched_preprocessor_init)

    return tmp_path


def _make_training_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X_train = pd.DataFrame(
        {
            "category": ["A", "B", "A", "C"],
            "value": [0.0, 1.0, 2.0, 3.0],
            "timestamp": [
                "2026-01-01T00:00:00Z",
                "2026-01-02T01:00:00Z",
                "2026-01-03T02:00:00Z",
                "2026-01-04T03:00:00Z",
            ],
        }
    )
    X_test = pd.DataFrame(
        {
            "category": ["D", "E"],
            "value": [4.0, np.nan],
            "timestamp": ["2026-01-05T04:00:00Z", "2026-01-06T05:00:00Z"],
        }
    )
    y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    y_test = np.array([4.0, 5.0], dtype=float)
    return X_train, X_test, y_train, y_test


def test_full_production_lifecycle(isolated_ml_artifacts: Path) -> None:
    """Train, persist, reload, and infer exactly as production should."""
    X_train, X_test, y_train, y_test = _make_training_data()

    training_service = TrainingService()
    result = training_service.train(X_train, X_test, y_train, y_test)

    assert result.saved_model.model_path.exists()
    assert result.saved_model.metadata_path.exists()
    assert result.registry_entry.artifact_path.exists()
    assert result.registry_entry.status == "Production"

    explainability_root = isolated_ml_artifacts / "ml" / "artifacts" / "explainability"
    assert explainability_root.exists(), "Explainability artifacts should be created during training"
    assert (explainability_root / result.best_model.model_name / str(result.saved_model.version) / "feature_importance.json").exists()
    assert (explainability_root / result.best_model.model_name / str(result.saved_model.version) / "metadata.json").exists()

    artifact_root = isolated_ml_artifacts
    metadata_path = artifact_root / "artifacts" / "preprocessing" / "preprocessor_metadata.json"
    assert metadata_path.exists(), "Preprocessor metadata should be persisted"

    onehot_path = artifact_root / "ml" / "models" / "preprocessing" / "onehot_encoder.pkl"
    ordinal_path = artifact_root / "ml" / "models" / "preprocessing" / "ordinal_encoder.pkl"
    scaler_path = artifact_root / "ml" / "models" / "preprocessing" / "standard_scaler.pkl"
    selector_path = artifact_root / "ml" / "models" / "preprocessing" / "feature_selection_model.pkl"

    assert onehot_path.exists() or ordinal_path.exists(), "At least one encoder should be persisted"
    assert scaler_path.exists(), "Scaler artifact should be persisted"
    assert selector_path.exists(), "Feature selector artifact should be persisted"

    persistence_engine = ModelPersistenceEngine()
    loaded_pipeline = persistence_engine.load_model(result.saved_model.model_path)
    assert hasattr(loaded_pipeline, "named_steps")
    assert "preprocessor" in loaded_pipeline.named_steps
    preprocessor = loaded_pipeline.named_steps["preprocessor"]
    assert preprocessor.output_feature_names_

    inference_input = pd.DataFrame(
        {
            "category": ["D", "E"],
            "value": [4.0, np.nan],
            "timestamp": ["2026-01-05T04:00:00Z", "2026-01-06T05:00:00Z"],
        }
    )

    prediction_engine = PredictionPipelineEngine()
    first_result = prediction_engine.predict(result.saved_model.model_path, inference_input)
    second_result = prediction_engine.predict(result.saved_model.model_path, inference_input.copy())

    assert np.array_equal(first_result.predictions, second_result.predictions)
    assert first_result.number_of_rows == 2
    assert first_result.model_version == result.saved_model.version

    transformed = preprocessor.transform(inference_input)
    assert list(transformed.columns) == preprocessor.output_feature_names_

    reordered = inference_input[["timestamp", "value", "category"]]
    reordered_result = prediction_engine.predict(result.saved_model.model_path, reordered)
    assert np.array_equal(first_result.predictions, reordered_result.predictions)

    with pytest.raises(ValueError, match="Input schema does not match training schema"):
        prediction_engine.predict(result.saved_model.model_path, inference_input.drop(columns=["category"]))

    with pytest.raises(ValueError, match="Input schema does not match training schema"):
        prediction_engine.predict(result.saved_model.model_path, pd.concat([inference_input, pd.DataFrame({"extra": [1, 2]})], axis=1))

    with pytest.raises(ValueError, match="Input data cannot be empty"):
        prediction_engine.predict(result.saved_model.model_path, pd.DataFrame(columns=["category", "value", "timestamp"]))


@pytest.mark.parametrize("input_data", [
    np.array([[4.0, "D", "2026-01-05T04:00:00Z"], [np.nan, "E", "2026-01-06T05:00:00Z"]]),
    pd.DataFrame(
        {
            "category": ["D", "E"],
            "value": [4.0, np.nan],
            "timestamp": ["2026-01-05T04:00:00Z", "2026-01-06T05:00:00Z"],
        }
    ),
])
def test_edge_case_inputs_are_handled(input_data: Any, isolated_ml_artifacts: Path) -> None:
    """Verify numpy arrays, dataframes, datetime values, NaN values, and unseen categories work."""
    X_train, X_test, y_train, y_test = _make_training_data()
    training_service = TrainingService()
    result = training_service.train(X_train, X_test, y_train, y_test)

    prediction_engine = PredictionPipelineEngine()
    result_obj = prediction_engine.predict(result.saved_model.model_path, input_data)

    assert result_obj.number_of_rows == 2
    assert result_obj.predictions.shape == (2,)
    assert result_obj.model_version == result.saved_model.version
