"""Unit tests for the training service orchestration layer."""

from __future__ import annotations

import numpy as np
import pytest

from ml.training.training_service import TrainingService, TrainingServiceResult


@pytest.fixture
def service() -> TrainingService:
    """Create a training service instance for tests."""
    return TrainingService()


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a compact regression dataset for workflow tests."""
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    X_test = np.array([[4.0], [5.0]], dtype=float)
    y_train = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    y_test = np.array([4.0, 5.0], dtype=float)
    return X_train, X_test, y_train, y_test


def test_complete_workflow(service: TrainingService, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    """The training service should orchestrate training, evaluation, and persistence."""
    result = service.train(*sample_data)

    assert isinstance(result, TrainingServiceResult)
    assert len(result.trained_models) >= 1
    assert len(result.evaluation_results) >= 1
    assert len(result.cross_validation_results) >= 1
    assert len(result.hyperparameter_results) >= 1
    assert result.saved_model.model_path.exists()
    assert result.experiment.model_name == result.best_model.model_name
    assert result.registry_entry.status == "Production"


def test_empty_data_raises_value_error(service: TrainingService) -> None:
    """The training service should reject empty datasets."""
    with pytest.raises(ValueError, match="empty"):
        service.train(np.array([]), np.array([]), np.array([]), np.array([]))


def test_failure_propagation(service: TrainingService) -> None:
    """Underlying engine errors should propagate through the service."""
    with pytest.raises(ValueError, match="empty"):
        service.train(np.array([[1.0]]), np.array([[2.0]]), np.array([]), np.array([3.0]))


def test_persistence_and_registry_update(service: TrainingService, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Successful workflow runs should persist a model and update the registry."""
    result = service.train(*sample_data)

    assert result.saved_model.model_path.exists()
    assert result.registry_entry.version == result.saved_model.version
    assert result.registry_entry.model_name == result.best_model.model_name


def test_experiment_created(service: TrainingService, sample_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    """The service should create an experiment record for the best model."""
    result = service.train(*sample_data)

    assert result.experiment.model_name == result.best_model.model_name
    assert result.experiment.model_version == result.saved_model.version
