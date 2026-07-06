"""Persist explainability artifacts for trained ETAIQ models."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None

from dataclasses import dataclass
from .explainability import ExplainabilityEngine, ExplanationResult

# Try to import real RegisteredModel from model_registry
try:
    from .model_registry import RegisteredModel
except ImportError:
    # Fallback to simple dataclass for compatibility without xgboost
    @dataclass
    class RegisteredModel:
        model_name: str
        version: int
        artifact_path: str
        metrics: dict
        created_at: str
        status: str
        metadata: dict


class ExplainabilityArtifactInconsistencyError(ValueError):
    """Raised when explainability metadata is inconsistent with model registry."""
    pass



class ExplainabilityArtifactGenerator:
    """Generate and persist explainability artifacts for a trained model."""

    def __init__(self, *, artifacts_root: str | Path | None = None) -> None:
        self.artifacts_root = Path(artifacts_root or Path("ml/artifacts/explainability")).resolve()
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def generate_for_model(
        self,
        model: Any,
        registry_entry: RegisteredModel,
        explanation: ExplanationResult | None = None,
        input_data: Any | None = None,
    ) -> dict[str, Any]:
        """Generate feature-importance and local explanation artifacts for a model and persist them to disk."""
        # Get model metadata exclusively from the registry entry
        model_name = registry_entry.model_name
        version = registry_entry.version
        metrics = dict(registry_entry.metrics)
        feature_names = list(registry_entry.metadata.get("feature_names", []))
        feature_count = registry_entry.metadata.get("feature_count", len(feature_names))
        target = registry_entry.metadata.get("target_column", "")
        
        if not feature_names:
            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            elif hasattr(model, "n_features_in_"):
                feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
            else:
                feature_names = ["feature_0"]
            feature_count = len(feature_names)

        engine = ExplainabilityEngine()
        result = explanation or engine.explain_model(model, model_name, feature_names, input_data=input_data)

        output_dir = self.artifacts_root / model_name / str(version or "latest")
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_importance_path = output_dir / "feature_importance.json"
        feature_importance_csv_path = output_dir / "feature_importance.csv"
        local_explanation_path = output_dir / "local_explanation.json"
        shap_summary_path = output_dir / "shap_summary.json"
        summary_plot_path = output_dir / "summary_plot.png"
        waterfall_plot_path = output_dir / "waterfall_plot.png"
        metadata_path = output_dir / "metadata.json"

        feature_importance_payload = {
            "model_name": model_name,
            "method": result.explanation_method,
            "feature_importance": result.feature_importance,
            "ranked_features": result.ranked_features,
            "confidence_score": result.confidence_score,
            "uncertainty_estimate": result.uncertainty_estimate,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        feature_importance_path.write_text(json.dumps(feature_importance_payload, indent=2), encoding="utf-8")

        local_explanation_payload = {
            "model_name": model_name,
            "local_explanation": result.local_explanation or [],
            "ranked_feature_impacts": result.ranked_feature_impacts or [],
            "confidence_score": result.confidence_score,
            "uncertainty_estimate": result.uncertainty_estimate,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        local_explanation_path.write_text(json.dumps(local_explanation_payload, indent=2), encoding="utf-8")

        with feature_importance_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["feature_name", "importance", "rank"])
            writer.writeheader()
            for index, item in enumerate(result.ranked_features, start=1):
                writer.writerow({
                    "feature_name": item.get("feature_name"),
                    "importance": item.get("importance"),
                    "rank": index,
                })

        shap_payload = {
            "model_name": model_name,
            "method": result.explanation_method,
            "summary": [
                {"feature_name": item.get("feature_name"), "importance": item.get("importance")}
                for item in result.ranked_features[:10]
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        shap_summary_path.write_text(json.dumps(shap_payload, indent=2), encoding="utf-8")

        self._write_plot(summary_plot_path, result.ranked_features, title=f"{model_name} feature importance")
        self._write_plot(waterfall_plot_path, result.ranked_feature_impacts or result.ranked_features, title=f"{model_name} feature impacts")

        metadata_payload = {
            "model_name": model_name,
            "version": version,
            "trained_at": registry_entry.created_at,
            "dataset": "eta_dataset",
            "feature_count": feature_count,
            "target": target,
            "framework": "xgboost" if "XGB" in model_name else "scikit-learn",
            "metrics": metrics,
            "feature_names": feature_names,
            "explainability_available": True,
            "feature_importance_path": str(feature_importance_path),
            "local_explanation_path": str(local_explanation_path),
            "shap_path": str(shap_summary_path),
            "summary_plot_path": str(summary_plot_path),
            "waterfall_plot_path": str(waterfall_plot_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Validation: Check consistency with registry entry
        validation_errors = []
        
        if metadata_payload["model_name"] != registry_entry.model_name:
            validation_errors.append(f"Model name mismatch: expected {registry_entry.model_name}, got {metadata_payload['model_name']}")
        if metadata_payload["version"] != registry_entry.version:
            validation_errors.append(f"Version mismatch: expected {registry_entry.version}, got {metadata_payload['version']}")
        if metadata_payload["feature_count"] != len(feature_names):
            validation_errors.append(f"Feature count mismatch: expected {len(feature_names)}, got {metadata_payload['feature_count']}")
        if metadata_payload["feature_names"] != list(feature_names):
            validation_errors.append(f"Feature names mismatch")
        if set(metadata_payload["metrics"].keys()) != set(registry_entry.metrics.keys()):
            validation_errors.append(f"Metrics keys mismatch")
            
        if validation_errors:
            raise ExplainabilityArtifactInconsistencyError("\n".join(validation_errors))
        
        metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

        return {
            "output_dir": str(output_dir),
            "feature_importance_path": str(feature_importance_path),
            "feature_importance_csv_path": str(feature_importance_csv_path),
            "local_explanation_path": str(local_explanation_path),
            "shap_summary_path": str(shap_summary_path),
            "summary_plot_path": str(summary_plot_path),
            "waterfall_plot_path": str(waterfall_plot_path),
            "metadata_path": str(metadata_path),
            "metadata": metadata_payload,
        }

    def _write_plot(self, output_path: Path, ranked_items: list[dict[str, Any]], *, title: str) -> None:
        """Write a simple bar chart placeholder to disk when matplotlib is available."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if plt is None:
            output_path.write_bytes(b"")
            return

        labels = [str(item.get("feature_name", "feature")) for item in ranked_items[:8]]
        values = [float(item.get("importance") or item.get("contribution_score") or 0.0) for item in ranked_items[:8]]
        figure, axis = plt.subplots(figsize=(6, 3))
        axis.bar(labels, values)
        axis.set_title(title)
        axis.set_ylabel("importance")
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(output_path, dpi=120)
        plt.close(figure)
