"""Unit tests for drift detection."""

from __future__ import annotations

import numpy as np
import pytest

from ml.training.drift_detection import DriftDetectionEngine, DriftResult


def test_fit_baseline_and_detect_no_drift() -> None:
    """Baseline fitting should support stable data with no drift detection."""
    engine = DriftDetectionEngine()
    X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    X_current = np.array([[1.1], [2.1], [3.1], [4.1]])

    engine.fit_baseline(X_train)
    result = engine.detect_drift(X_current)

    assert len(result) == 1
    assert isinstance(result[0], DriftResult)
    assert result[0].drift_detected is False


def test_detect_drift_when_distribution_shifts() -> None:
    """A large shift in the feature distribution should be flagged."""
    engine = DriftDetectionEngine()
    engine.fit_baseline(np.array([[1.0], [2.0], [3.0], [4.0]]))
    result = engine.detect_drift(np.array([[10.0], [11.0], [12.0], [13.0]]))

    assert result[0].drift_detected is True


def test_validation_errors() -> None:
    """Validation should reject empty inputs and feature mismatches."""
    engine = DriftDetectionEngine()

    with pytest.raises(ValueError, match="empty baseline"):
        engine.fit_baseline(np.array([]))

    with pytest.raises(ValueError, match="empty current"):
        engine.detect_drift(np.array([]))

    with pytest.raises(ValueError, match="feature mismatch"):
        engine.fit_baseline(np.array([[1.0], [2.0]]))
        engine.detect_drift(np.array([[1.0, 2.0]]))
