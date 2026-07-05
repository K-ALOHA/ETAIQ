"""Unit tests for the model registry engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.training.model_registry import ModelRegistryEngine, RegisteredModel


def test_register_and_lookup_model(tmp_path: Path) -> None:
    """Models should be registered and retrievable by name and version."""
    engine = ModelRegistryEngine(storage_dir=tmp_path)
    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_bytes(b"model")

    registered = engine.register_model(
        model_name="LinearRegression",
        version=1,
        artifact_path=artifact_path,
        metrics={"mae": 0.1},
        status="Staging",
    )

    assert isinstance(registered, RegisteredModel)
    assert engine.get_model("LinearRegression", 1).version == 1
    assert len(engine.list_models()) == 1


def test_set_production_archives_previous_model(tmp_path: Path) -> None:
    """Promoting a model should archive the previous production model."""
    engine = ModelRegistryEngine(storage_dir=tmp_path)
    first_artifact = tmp_path / "first.joblib"
    second_artifact = tmp_path / "second.joblib"
    first_artifact.write_bytes(b"first")
    second_artifact.write_bytes(b"second")

    engine.register_model("LinearRegression", 1, first_artifact, metrics={"mae": 0.1}, status="Production")
    engine.register_model("LinearRegression", 2, second_artifact, metrics={"mae": 0.05}, status="Staging")

    engine.set_production("LinearRegression", 2)

    first = engine.get_model("LinearRegression", 1)
    second = engine.get_model("LinearRegression", 2)
    assert first.status == "Archived"
    assert second.status == "Production"


def test_unknown_model_raises_value_error(tmp_path: Path) -> None:
    """Operations on unknown model names should raise ValueError."""
    engine = ModelRegistryEngine(storage_dir=tmp_path)

    with pytest.raises(ValueError, match="unknown model"):
        engine.get_model("MissingModel", 1)


def test_missing_artifact_raises_value_error(tmp_path: Path) -> None:
    """Registering a missing artifact should raise ValueError."""
    engine = ModelRegistryEngine(storage_dir=tmp_path)

    with pytest.raises(ValueError, match="missing artifact"):
        engine.register_model("LinearRegression", 1, tmp_path / "missing.joblib", metrics={}, status="Staging")


def test_archive_model_updates_status(tmp_path: Path) -> None:
    """Archiving a model should set its status to Archived."""
    engine = ModelRegistryEngine(storage_dir=tmp_path)
    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_bytes(b"model")

    registered = engine.register_model("LinearRegression", 1, artifact_path, metrics={}, status="Staging")
    engine.archive_model("LinearRegression", 1)

    assert engine.get_model("LinearRegression", 1).status == "Archived"
    assert registered.status == "Staging"


def test_update_explainability_artifacts_metadata(tmp_path: Path) -> None:
    """Registry metadata should accept explainability artifact locations for production models."""
    engine = ModelRegistryEngine(storage_dir=tmp_path)
    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_bytes(b"model")

    registered = engine.register_model("LinearRegression", 1, artifact_path, metrics={}, status="Production")
    updated = engine.update_explainability_metadata(
        model_name="LinearRegression",
        version=1,
        explainability_dir=tmp_path / "explainability",
        feature_importance_path=tmp_path / "explainability" / "feature_importance.json",
        local_explanation_path=tmp_path / "explainability" / "local_explanation.json",
        metadata_path=tmp_path / "explainability" / "metadata.json",
    )

    assert updated.metadata["explainability_available"] is True
    assert updated.metadata["explainability_dir"] == str(tmp_path / "explainability")
    assert updated.metadata["feature_importance_path"] == str(tmp_path / "explainability" / "feature_importance.json")
