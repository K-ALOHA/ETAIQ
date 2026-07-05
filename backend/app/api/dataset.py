"""Dataset management API routes for ETAIQ."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.ml_management import DatasetResponse

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(tags=["dataset"])


@router.get(
    "/dataset",
    response_model=DatasetResponse,
    summary="Get dataset summary",
    description="Returns summary of the training dataset.",
)
async def get_dataset() -> DatasetResponse:
    """Return dataset summary."""
    logger.info("dataset_requested", endpoint="/api/v1/dataset")

    try:
        repo_root = Path(__file__).resolve().parents[3]
        dataset_path = repo_root / "ml" / "data" / "features" / "engineered_training_dataset.csv"
        if not dataset_path.exists():
            return DatasetResponse(
                record_count=0,
                feature_count=0,
                target_column=None,
                missing_values={},
                feature_names=[],
            )
        
        dataframe = pd.read_csv(dataset_path)
        missing_summary = dataframe.isna().sum().to_dict()
        missing_values = {k: int(v) for k, v in missing_summary.items() if v > 0}
        
        return DatasetResponse(
            record_count=int(len(dataframe)),
            feature_count=len(dataframe.columns),
            target_column="actual_delivery_time_min",
            missing_values=missing_values,
            feature_names=[str(col) for col in dataframe.columns],
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("dataset_failed", endpoint="/api/v1/dataset", error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="unable to get dataset summary") from exc
