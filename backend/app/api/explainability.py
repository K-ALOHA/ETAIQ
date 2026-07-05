"""Explainability management API routes for ETAIQ."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, status

from app.api.models import registry_engine
from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.ml_management import ExplainabilityLatestResponse, ExplainabilityResponse
from ml.training.explainability import ExplainabilityEngine
from ml.training.explainability_artifacts import ExplainabilityArtifactGenerator
from ml.training.monitoring import MonitoringEngine
from ml.training.persistence import ModelPersistenceEngine
from ml.training.model_registry import RegisteredModel

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(tags=["explainability"])


@router.get(
    "/explainability/latest",
    response_model=ExplainabilityLatestResponse,
    summary="Get the latest persisted explainability workspace payload",
    description="Returns explainability artifacts and summary metrics for the latest production model.",
)
async def get_latest_explainability() -> ExplainabilityLatestResponse:
    """Return the latest explainability payload for the active production model."""
    logger.info("latest_explainability_requested", endpoint="/api/v1/explainability/latest")

    repo_root = Path(__file__).resolve().parents[3]
    production_model = _select_latest_production_model()
    if production_model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No Production XGBRegressor model is registered")

    # Try the fast path: locate a pre-existing artifact directory.
    # When _find_latest_artifact_dir is monkeypatched (e.g. in tests) it may
    # return None, in which case we fall through to on-demand generation.
    prebuilt_dir = _find_latest_artifact_dir(production_model.model_name)
    if prebuilt_dir is not None:
        artifact_context: dict[str, Any] | None = {
            "output_dir": str(prebuilt_dir),
            "metadata_path": str(prebuilt_dir / "metadata.json"),
        }
    else:
        artifact_context = _ensure_explainability_artifacts(production_model, repo_root, force_regenerate=True)
    artifact_dir = Path(artifact_context["output_dir"]) if artifact_context else None
    if artifact_dir is None or not artifact_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No explainability artifacts found")

    metadata_path = artifact_dir / "metadata.json"
    feature_importance_path = artifact_dir / "feature_importance.json"
    local_explanation_path = artifact_dir / "local_explanation.json"
    summary_plot_path = artifact_dir / "summary_plot.png"
    waterfall_plot_path = artifact_dir / "waterfall_plot.png"

    metadata_payload = _read_json_file(metadata_path) or {}
    feature_payload = _read_json_file(feature_importance_path) or {}
    local_payload = _read_json_file(local_explanation_path) or {}

    monitoring_engine = MonitoringEngine(load_existing_records=True)
    latest_monitor_record = monitoring_engine.get_latest()

    latest_prediction_value = latest_monitor_record.mean_prediction if latest_monitor_record else 0.0
    prediction_time = latest_monitor_record.timestamp if latest_monitor_record else str(metadata_payload.get("trained_at") or metadata_payload.get("generated_at") or "")
    confidence_score = float(
        feature_payload.get("confidence_score")
        if feature_payload.get("confidence_score") is not None
        else local_payload.get("confidence_score", 0.0)
    )

    return ExplainabilityLatestResponse(
        model_name=str(feature_payload.get("model_name") or metadata_payload.get("model_name") or production_model.model_name),
        version=str(metadata_payload.get("version") or production_model.version),
        latest_prediction_value=float(latest_prediction_value),
        prediction_time=str(prediction_time),
        confidence_score=confidence_score,
        explainability_status="ready" if metadata_payload.get("explainability_available") else "unavailable",
        feature_importance=feature_payload.get("feature_importance") or {},
        ranked_features=feature_payload.get("ranked_features") or [],
        local_explanation=local_payload.get("local_explanation") or [],
        natural_language_explanation=_build_natural_language_explanation(local_payload),
        summary_plot=_read_image_data(summary_plot_path),
        waterfall_plot=_read_image_data(waterfall_plot_path),
        plots={
            "summary_plot": _read_image_data(summary_plot_path),
            "waterfall_plot": _read_image_data(waterfall_plot_path),
        },
        metadata_json=json.dumps(metadata_payload, indent=2),
        metadata=metadata_payload,
    )


@router.get(
    "/explainability/feature-importance",
    summary="Get feature importance for production model",
    description="Returns feature importance data for the active production model.",
)
async def get_feature_importance():
    """Return feature importance for the active production model."""
    logger.info("feature_importance_requested", endpoint="/api/v1/explainability/feature-importance")

    production_model = _select_latest_production_model()
    if production_model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No Production XGBRegressor model is registered")

    artifact_context = _ensure_explainability_artifacts(production_model, Path(__file__).resolve().parents[3])
    artifact_dir = Path(artifact_context["output_dir"]) if artifact_context else None
    if artifact_dir is None or not artifact_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No explainability artifacts found")

    feature_importance_path = artifact_dir / "feature_importance.json"
    feature_payload = _read_json_file(feature_importance_path) or {}

    return {
        "model_name": feature_payload.get("model_name", production_model.model_name),
        "version": production_model.version,
        "feature_importance": feature_payload.get("feature_importance", {}),
        "ranked_features": feature_payload.get("ranked_features", []),
        "method": feature_payload.get("method", "persisted_artifact"),
        "generated_at": feature_payload.get("generated_at"),
    }


@router.get(
    "/explainability/local",
    summary="Get local explanation for production model",
    description="Returns local explanation data for the active production model.",
)
async def get_local_explanation():
    """Return local explanation for the active production model."""
    logger.info("local_explanation_requested", endpoint="/api/v1/explainability/local")

    production_model = _select_latest_production_model()
    if production_model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No Production XGBRegressor model is registered")

    artifact_context = _ensure_explainability_artifacts(production_model, Path(__file__).resolve().parents[3])
    artifact_dir = Path(artifact_context["output_dir"]) if artifact_context else None
    if artifact_dir is None or not artifact_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No explainability artifacts found")

    local_explanation_path = artifact_dir / "local_explanation.json"
    local_payload = _read_json_file(local_explanation_path) or {}
    feature_importance_path = artifact_dir / "feature_importance.json"
    feature_payload = _read_json_file(feature_importance_path) or {}

    return {
        "model_name": local_payload.get("model_name", production_model.model_name),
        "version": production_model.version,
        "local_explanation": local_payload.get("local_explanation", []),
        "ranked_feature_impacts": local_payload.get("ranked_feature_impacts", []),
        "confidence_score": local_payload.get("confidence_score", 0.0),
        "uncertainty_estimate": local_payload.get("uncertainty_estimate", 0.0),
        "generated_at": local_payload.get("generated_at"),
        "feature_importance": feature_payload.get("feature_importance", {}),
        "ranked_features": feature_payload.get("ranked_features", []),
    }


@router.get(
    "/explainability/shap",
    summary="Get SHAP explanation for production model",
    description="Returns SHAP explanation data for the active production model.",
)
async def get_shap_explanation():
    """Return SHAP explanation for the active production model."""
    logger.info("shap_explanation_requested", endpoint="/api/v1/explainability/shap")

    production_model = _select_latest_production_model()
    if production_model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No Production XGBRegressor model is registered")

    artifact_context = _ensure_explainability_artifacts(production_model, Path(__file__).resolve().parents[3])
    artifact_dir = Path(artifact_context["output_dir"]) if artifact_context else None
    if artifact_dir is None or not artifact_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No explainability artifacts found")

    shap_path = artifact_dir / "shap_summary.json"
    shap_payload = _read_json_file(shap_path) or {}
    feature_importance_path = artifact_dir / "feature_importance.json"
    feature_payload = _read_json_file(feature_importance_path) or {}

    return {
        "model_name": shap_payload.get("model_name", production_model.model_name),
        "version": production_model.version,
        "method": shap_payload.get("method", "persisted_artifact"),
        "summary": shap_payload.get("summary", []),
        "generated_at": shap_payload.get("generated_at"),
        "feature_importance": feature_payload.get("feature_importance", {}),
        "ranked_features": feature_payload.get("ranked_features", []),
    }


@router.get(
    "/explainability/{model_name}",
    response_model=ExplainabilityResponse,
    summary="Get explainability output for a model",
    description="Returns explainability output derived from the production model registry.",
)
async def get_explainability(model_name: str) -> ExplainabilityResponse:
    """Return explainability output for the requested production model."""
    logger.info("explainability_requested", endpoint="/api/v1/explainability", model_name=model_name)

    try:
        production_model = registry_engine.select_production_model("XGBRegressor")
        repo_root = Path(__file__).resolve().parents[3]
        artifact_context = _ensure_explainability_artifacts(production_model, repo_root)
        artifact_dir = Path(artifact_context["output_dir"]) if artifact_context else None
        if artifact_dir is None or not artifact_dir.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No explainability artifacts found")

        metadata_path = artifact_dir / "metadata.json"
        feature_importance_path = artifact_dir / "feature_importance.json"
        metadata_payload = _read_json_file(metadata_path) or {}
        feature_payload = _read_json_file(feature_importance_path) or {}
        if metadata_payload or feature_payload:
            return ExplainabilityResponse(
                model_name=str(feature_payload.get("model_name") or metadata_payload.get("model_name") or model_name),
                feature_importance=feature_payload.get("feature_importance") or {},
                ranked_features=feature_payload.get("ranked_features") or [],
                explanation_time_seconds=0.0,
                explanation_method=feature_payload.get("method") or metadata_payload.get("framework") or "persisted_artifact",
                sample_count=len(feature_payload.get("feature_importance") or {}),
            )

        model = ModelPersistenceEngine().load_model(production_model.artifact_path)
        engine = ExplainabilityEngine()
        feature_names = _resolve_feature_names(production_model)
        result = engine.explain_model(model, model_name, feature_names or ["feature_0"])
        return ExplainabilityResponse(
            model_name=result.model_name,
            feature_importance=result.feature_importance,
            ranked_features=result.ranked_features,
            explanation_time_seconds=result.explanation_time_seconds,
            explanation_method=result.explanation_method,
            sample_count=result.sample_count,
        )
    except FileNotFoundError as exc:
        logger.error("model_not_found", endpoint="/api/v1/explainability", error=str(exc))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        logger.error("explainability_no_xgb_model", endpoint="/api/v1/explainability", error=str(exc))
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("explainability_failed", endpoint="/api/v1/explainability", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="unable to generate explainability") from exc


def _find_latest_artifact_dir(model_name: str) -> Path | None:
    """Return the latest persisted explainability artifact directory for a model, or None."""
    repo_root = Path(__file__).resolve().parents[3]
    artifact_root = repo_root / "ml" / "artifacts" / "explainability" / model_name
    if not artifact_root.exists():
        return None
    version_dirs = sorted(
        (d for d in artifact_root.iterdir() if d.is_dir()),
        key=lambda d: int(d.name) if d.name.isdigit() else 0,
        reverse=True,
    )
    for version_dir in version_dirs:
        if (version_dir / "metadata.json").exists():
            return version_dir
    return None


def _select_latest_production_model() -> RegisteredModel | None:
    """Select the latest Production XGBRegressor model from the registry."""
    try:
        return registry_engine.select_production_model("XGBRegressor")
    except ValueError:
        return None


def _read_json_file(file_path: Path) -> dict[str, Any] | None:
    """Read a JSON file and return its contents, or None if it doesn't exist or can't be read."""
    try:
        if file_path.exists():
            return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _read_image_data(file_path: Path) -> str | None:
    """Read an image file and return it as a base64-encoded string, or None if it doesn't exist or can't be read."""
    try:
        if file_path.exists():
            return base64.b64encode(file_path.read_bytes()).decode("utf-8")
    except Exception:
        pass
    return None


def _build_natural_language_explanation(local_payload: dict[str, Any]) -> str:
    """Build a natural language explanation from local explanation data."""
    try:
        local_explanation = local_payload.get("local_explanation", [])
        if not local_explanation:
            return "No local explanation available."
        
        ranked = sorted(
            local_explanation,
            key=lambda x: abs(float(x.get("contribution_score", 0))),
            reverse=True
        )[:3]
        
        if not ranked:
            return "No significant feature contributions found."
        
        feature_names = [item.get("feature_name", "").replace("_", " ") for item in ranked]
        contributions = [float(item.get("contribution_score", 0)) for item in ranked]
        
        parts = []
        for name, contrib in zip(feature_names, contributions):
            direction = "increasing" if contrib > 0 else "decreasing"
            parts.append(f"{name} ({direction})")
        
        return f"The prediction is primarily influenced by {', and '.join(parts)}."
    except Exception:
        return "Failed to generate natural language explanation."


def _resolve_feature_names(production_model: RegisteredModel) -> list[str]:
    """Resolve feature names from a registered model's metadata."""
    try:
        metadata = getattr(production_model, "metadata", {}) or {}
        feature_names = metadata.get("feature_names", [])
        if feature_names:
            return list(feature_names)
    except Exception:
        pass
    return []


def _ensure_explainability_artifacts(production_model: RegisteredModel, repo_root: Path, *, force_regenerate: bool = False) -> dict[str, Any] | None:
    """Reuse or generate explainability artifacts for the active production model."""
    artifact_root = repo_root / "ml" / "artifacts" / "explainability"
    metadata_payload = dict(getattr(production_model, "metadata", {}) or {})

    if not force_regenerate:
        metadata_path = None
        registry_metadata_path = metadata_payload.get("metadata_path")
        if registry_metadata_path:
            resolved_metadata_path = Path(registry_metadata_path)
            if not resolved_metadata_path.is_absolute():
                resolved_metadata_path = (repo_root / resolved_metadata_path).resolve()
            if resolved_metadata_path.exists():
                metadata_path = resolved_metadata_path

        feature_importance_path = None
        registry_feature_importance_path = metadata_payload.get("feature_importance_path")
        if registry_feature_importance_path:
            resolved_feature_importance_path = Path(registry_feature_importance_path)
            if not resolved_feature_importance_path.is_absolute():
                resolved_feature_importance_path = (repo_root / resolved_feature_importance_path).resolve()
            if resolved_feature_importance_path.exists():
                feature_importance_path = resolved_feature_importance_path

        if metadata_path is not None:
            return {
                "output_dir": str(metadata_path.parent),
                "metadata_path": str(metadata_path),
            }

        if feature_importance_path is not None:
            return {
                "output_dir": str(feature_importance_path.parent),
                "metadata_path": str(feature_importance_path.parent / "metadata.json"),
            }

        expected_dir = artifact_root / production_model.model_name / str(production_model.version)
        fallback_metadata_path = expected_dir / "metadata.json"
        if fallback_metadata_path.exists():
            return {
                "output_dir": str(expected_dir),
                "metadata_path": str(fallback_metadata_path),
            }

    persistence_engine = ModelPersistenceEngine()
    model = persistence_engine.load_model(production_model.artifact_path)
    generator = ExplainabilityArtifactGenerator(artifacts_root=artifact_root)
    return generator.generate_for_model(
        model,
        production_model,
    )
