"""Drift detection utilities for ETAIQ model monitoring."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .logging_config import TrainingLogger


@dataclass
class DriftResult:
    """Drift statistics for a single feature."""

    feature_name: str
    baseline_mean: float
    current_mean: float
    baseline_std: float
    current_std: float
    drift_score: float
    drift_detected: bool


class DriftDetectionEngine:
    """Measure feature drift between a baseline and current dataset."""

    def __init__(self, logger: TrainingLogger | None = None, threshold: float = 0.5) -> None:
        self._logger = logger or TrainingLogger(name="training.drift_detection")
        self._baseline: np.ndarray | None = None
        self._feature_names: list[str] = []
        self._threshold = threshold

    def fit_baseline(self, X_train: Any) -> None:
        """Store a baseline feature matrix for later drift checks."""
        baseline = self._validate_feature_matrix(X_train, kind="baseline")
        self._baseline = np.asarray(baseline, dtype=float)
        self._feature_names = [str(index) for index in range(self._baseline.shape[1])]
        self._logger.info("Baseline fitted", rows=self._baseline.shape[0], features=self._baseline.shape[1])

    def detect_drift(self, X_current: Any) -> list[DriftResult]:
        """Compare current data against the baseline and flag drift."""
        current = self._validate_feature_matrix(X_current, kind="current")
        current_array = np.asarray(current, dtype=float)

        if self._baseline is None:
            raise ValueError("baseline must be fitted before detecting drift")
        if current_array.shape[1] != self._baseline.shape[1]:
            raise ValueError("feature mismatch between baseline and current data")

        results: list[DriftResult] = []
        for feature_index in range(self._baseline.shape[1]):
            baseline_values = self._baseline[:, feature_index]
            current_values = current_array[:, feature_index]
            baseline_mean = float(baseline_values.mean())
            current_mean = float(current_values.mean())
            baseline_std = float(baseline_values.std())
            current_std = float(current_values.std())
            drift_score = abs(current_mean - baseline_mean) / max(baseline_std, 1e-8)
            drift_detected = drift_score > self._threshold
            results.append(
                DriftResult(
                    feature_name=self._feature_names[feature_index],
                    baseline_mean=baseline_mean,
                    current_mean=current_mean,
                    baseline_std=baseline_std,
                    current_std=current_std,
                    drift_score=drift_score,
                    drift_detected=drift_detected,
                )
            )

        self._logger.info("Drift detected", drift_count=sum(1 for result in results if result.drift_detected))
        return results

    def export_report(self, output_path: str | Path) -> Path:
        """Export the latest drift report to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(result) for result in self.detect_drift(self._baseline)] if self._baseline is not None else "[]", indent=2), encoding="utf-8")
        self._logger.info("Drift report exported", output_path=str(path))
        return path

    def _validate_feature_matrix(self, X: Any, kind: str) -> np.ndarray:
        """Validate and normalize a feature matrix for drift analysis."""
        if X is None:
            raise ValueError(f"{kind} data cannot be None")

        array = np.asarray(X)
        if array.size == 0:
            raise ValueError(f"empty {kind} data")

        if array.ndim == 1:
            array = array.reshape(-1, 1)

        if array.ndim != 2:
            raise ValueError(f"{kind} data must be a 2D array")

        return array
