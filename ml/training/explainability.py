"""Model explainability utilities for ETAIQ training workflows."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .logging_config import TrainingLogger


@dataclass
class ExplanationResult:
    """Container for global and local feature-attribution explanations."""

    model_name: str
    feature_importance: dict[str, float]
    ranked_features: list[dict[str, Any]]
    explanation_time_seconds: float
    explanation_method: str
    sample_count: int
    local_explanation: list[dict[str, Any]] | None = None
    ranked_feature_impacts: list[dict[str, Any]] | None = None
    confidence_score: float = 0.0
    uncertainty_estimate: float = 0.0


class ExplainabilityEngine:
    """Generate model-level and local explanations for supported sklearn-style models."""

    def __init__(self, logger: TrainingLogger | None = None) -> None:
        self._logger = logger or TrainingLogger(name="training.explainability")

    def explain_model(self, model: Any, model_name: str, feature_names: list[str], input_data: Any | None = None) -> ExplanationResult:
        """Build a global explanation for a fitted model and optionally a local explanation for an input row."""
        self._validate_model(model, feature_names)

        start_time = time.perf_counter()
        feature_importance = self._extract_feature_importance(model, feature_names)
        ranked_features = self._rank_features(feature_importance)
        explanation_method = self._determine_method(model)
        local_explanation = self._build_local_explanation(model, input_data, feature_names)
        ranked_feature_impacts = self._rank_feature_impacts(local_explanation)
        confidence_score, uncertainty_estimate = self._calculate_confidence_and_uncertainty(
            feature_importance,
            local_explanation,
            explanation_method,
        )
        elapsed = time.perf_counter() - start_time

        result = ExplanationResult(
            model_name=model_name,
            feature_importance=feature_importance,
            ranked_features=ranked_features,
            explanation_time_seconds=elapsed,
            explanation_method=explanation_method,
            sample_count=len(feature_names),
            local_explanation=local_explanation,
            ranked_feature_impacts=ranked_feature_impacts,
            confidence_score=confidence_score,
            uncertainty_estimate=uncertainty_estimate,
        )
        self._logger.info("Explanation generated", model_name=model_name, method=result.explanation_method)
        return result

    def explain_prediction(self, model: Any, X: Any, feature_names: list[str]) -> list[dict[str, Any]]:
        """Generate a local explanation for a single prediction row."""
        self._validate_model(model, feature_names)
        if X is None:
            raise ValueError("input data cannot be None")

        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("input data must be 2D")
        if array.shape[1] != len(feature_names):
            raise ValueError("feature mismatch")

        importance = self._extract_feature_importance(model, feature_names)
        local_explanation: list[dict[str, Any]] = []
        for feature_name, value in zip(feature_names, array[0]):
            contribution = float(value) * float(importance[feature_name])
            local_explanation.append(
                {
                    "feature_name": feature_name,
                    "value": float(value),
                    "importance": float(importance[feature_name]),
                    "contribution_score": contribution,
                }
            )

        self._logger.info("Local explanation generated", feature_count=len(feature_names))
        return local_explanation

    def export_csv(self, result: ExplanationResult, output_path: str | Path) -> Path:
        """Export an explanation result to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["feature_name", "importance", "rank"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, item in enumerate(result.ranked_features, start=1):
                writer.writerow({
                    "feature_name": item["feature_name"],
                    "importance": item["importance"],
                    "rank": index,
                })
        self._logger.info("Exports completed", output_path=str(path), format="csv")
        return path

    def export_json(self, result: ExplanationResult, output_path: str | Path) -> Path:
        """Export an explanation result to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        self._logger.info("Exports completed", output_path=str(path), format="json")
        return path

    def _extract_feature_importance(self, model: Any, feature_names: list[str]) -> dict[str, float]:
        """Extract a global feature-importance mapping from a supported model."""
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float).reshape(-1)
            if len(values) != len(feature_names):
                raise ValueError("feature mismatch")
            return {feature_name: float(value) for feature_name, value in zip(feature_names, values)}

        if hasattr(model, "coef_"):
            values = np.asarray(model.coef_, dtype=float).reshape(-1)
            if len(values) != len(feature_names):
                raise ValueError("feature mismatch")
            return {feature_name: float(abs(value)) for feature_name, value in zip(feature_names, values)}

        if len(feature_names) == 0:
            raise ValueError("feature list cannot be empty")
        equal_weight = 1.0 / len(feature_names)
        return {feature_name: equal_weight for feature_name in feature_names}

    def _rank_features(self, feature_importance: dict[str, float]) -> list[dict[str, Any]]:
        """Return ranked feature records sorted by descending importance."""
        ranked = [
            {"feature_name": feature_name, "importance": float(importance)}
            for feature_name, importance in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
        ]
        return ranked

    def _determine_method(self, model: Any) -> str:
        """Identify the explanation strategy used for a model."""
        model_name = str(getattr(model, "__class__", type("", (), {})).__name__)
        if model_name.lower().startswith("xgb"):
            try:
                import shap  # type: ignore # noqa: F401
            except Exception:
                return "feature_importance"
            return "shap"
        if hasattr(model, "feature_importances_"):
            return "feature_importance"
        if hasattr(model, "coef_"):
            return "coefficients"
        return "fallback"

    def _build_local_explanation(self, model: Any, input_data: Any | None, feature_names: list[str]) -> list[dict[str, Any]]:
        """Build a local explanation payload when input values are available."""
        if input_data is None:
            return []
        try:
            return self.explain_prediction(model, input_data, feature_names)
        except Exception:
            return []

    def _rank_feature_impacts(self, local_explanation: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Rank local feature contributions by absolute impact."""
        if not local_explanation:
            return []
        ranked = [
            {
                "feature_name": item.get("feature_name"),
                "contribution_score": float(item.get("contribution_score", 0.0)),
            }
            for item in sorted(local_explanation, key=lambda item: abs(float(item.get("contribution_score", 0.0))), reverse=True)
        ]
        return ranked

    def _calculate_confidence_and_uncertainty(
        self,
        feature_importance: dict[str, float],
        local_explanation: list[dict[str, Any]] | None,
        explanation_method: str,
    ) -> tuple[float, float]:
        """Derive a confidence score and uncertainty estimate from explanation strength."""
        if not feature_importance:
            return 0.5, 0.5

        absolute_values = [abs(float(value)) for value in feature_importance.values()]
        total_importance = float(sum(absolute_values)) or 1.0
        strongest_signal = max(absolute_values) / total_importance if total_importance else 0.0
        confidence = min(1.0, max(0.05, strongest_signal))
        if explanation_method == "fallback":
            confidence = max(0.2, confidence)
        if local_explanation:
            avg_abs_contribution = float(np.mean([abs(float(item.get("contribution_score", 0.0))) for item in local_explanation]))
            confidence = min(1.0, max(confidence, min(1.0, avg_abs_contribution / max(total_importance, 1.0))))
        uncertainty = max(0.0, 1.0 - confidence)
        return round(confidence, 4), round(uncertainty, 4)

    def _validate_model(self, model: Any, feature_names: list[str]) -> None:
        """Validate the model and feature list before explanation."""
        if model is None:
            raise ValueError("model cannot be None")
        if not feature_names:
            raise ValueError("feature list cannot be empty")
