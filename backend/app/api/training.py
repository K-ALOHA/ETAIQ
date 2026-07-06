"""Training management API routes for ETAIQ."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.experiments import experiment_engine
from app.api.models import registry_engine
from app.core.logging import get_logger
from app.schemas.ml_management import TrainingRequest, TrainingResponse
from ml.training.training_service import TrainingService

logger = get_logger(__name__)
router = APIRouter(tags=["training"])


@router.post(
    "/train",
    response_model=TrainingResponse,
    summary="Train the available model registry workflow",
    description="Runs the existing training service and returns the selected model metadata.",
)
async def train(request: TrainingRequest) -> TrainingResponse:
    """Execute the training service and return metadata about the resulting model."""
    logger.info("training_requested", endpoint="/api/v1/train")

    try:
        service = TrainingService()
        result = service.train(request.X_train, request.X_test, request.y_train, request.y_test)

        try:
            # Extract all metrics from the comparison result for the best model
            best_metrics_entry = next(
                entry for entry in result.comparison_result.leaderboard
                if entry["model_name"] == result.best_model.model_name
            )
            all_metrics = {
                "mae": float(best_metrics_entry["mae"]),
                "rmse": float(best_metrics_entry["rmse"]),
                "r2": float(best_metrics_entry["r2"]),
                "mape": float(best_metrics_entry["mape"]),
            }
            
            registry_engine.register_model(
                model_name=result.best_model.model_name,
                version=result.saved_model.version,
                artifact_path=result.saved_model.model_path,
                metrics=all_metrics,
                status="Production",
            )
            registry_engine.set_production(result.best_model.model_name, result.saved_model.version)
        except ValueError:
            pass

        try:
            # Extract all metrics for experiment logging
            best_metrics_entry = next(
                entry for entry in result.comparison_result.leaderboard
                if entry["model_name"] == result.best_model.model_name
            )
            all_metrics = {
                "mae": float(best_metrics_entry["mae"]),
                "rmse": float(best_metrics_entry["rmse"]),
                "r2": float(best_metrics_entry["r2"]),
                "mape": float(best_metrics_entry["mape"]),
            }
        except (StopIteration, KeyError):
            all_metrics = {"mae": float(result.best_model.metric_value)}
        
        experiment_engine.log_experiment(
            model_name=result.best_model.model_name,
            dataset_version="default",
            hyperparameters={"selected_model": result.best_model.model_name},
            metrics=all_metrics,
            training_time_seconds=result.training_time,
            model_version=result.saved_model.version,
        )

        return TrainingResponse(
            best_model_name=result.best_model.model_name,
            best_model_version=result.saved_model.version,
            training_time_seconds=result.training_time,
            saved_model_path=str(result.saved_model.model_path),
            registry_status=result.registry_entry.status,
            experiment_id=result.experiment.experiment_id,
        )
    except ValueError as exc:
        logger.error("training_failed", endpoint="/api/v1/train", error=str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("training_failed", endpoint="/api/v1/train", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="training failed"
        ) from exc
