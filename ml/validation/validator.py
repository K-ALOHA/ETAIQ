"""Base validator protocol and validation engine orchestration."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from ml.validation.logging_config import get_logger
from ml.validation.models import ValidationResult, ValidationSummary
from ml.validation.quality_score import QualityScoreCalculator
from ml.validation.schemas import ORDERS_SCHEMA, DatasetSchema

logger = get_logger(__name__)

CHUNK_SIZE = 100_000


def load_dataframe(path: Path, chunksize: int | None = None) -> pd.DataFrame | Any:
    """Load a CSV file, optionally as an iterator of chunks.

    Args:
        path: Path to the CSV file.
        chunksize: When set, return a chunked reader instead of a DataFrame.

    Returns:
        pd.DataFrame or pandas TextFileReader when chunking is enabled.
    """
    if chunksize:
        return pd.read_csv(path, chunksize=chunksize, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def count_rows(path: Path, chunksize: int = CHUNK_SIZE) -> int:
    """Count rows in a CSV without loading the full file into memory.

    Args:
        path: Path to the CSV file.
        chunksize: Number of rows per read chunk.

    Returns:
        int: Total row count.
    """
    total = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        total += len(chunk)
    return total


class BaseValidator(ABC):
    """Abstract base class for dataset validators with structured logging."""

    name: str

    def run(
        self, df: pd.DataFrame, schema: DatasetSchema, **context: Any
    ) -> ValidationResult:
        """Execute validation with start/end logging and timing.

        Args:
            df: Dataset to validate.
            schema: Expected schema for the dataset.
            **context: Additional validator-specific context.

        Returns:
            ValidationResult: Outcome of the validation step.
        """
        logger.info(
            "validation_step_start",
            validator=self.name,
            dataset=schema.name,
            rows=len(df),
        )
        start = time.perf_counter()
        result = self.validate(df, schema, **context)
        duration = time.perf_counter() - start
        final = ValidationResult(
            validator_name=result.validator_name,
            dataset_name=result.dataset_name,
            passed=result.passed,
            score=result.score,
            details=result.details,
            duration_seconds=duration,
        )
        logger.info(
            "validation_step_end",
            validator=self.name,
            dataset=schema.name,
            passed=final.passed,
            score=round(final.score, 2),
            duration_seconds=round(duration, 4),
            result="pass" if final.passed else "fail",
        )
        return final

    @abstractmethod
    def validate(
        self, df: pd.DataFrame, schema: DatasetSchema, **context: Any
    ) -> ValidationResult:
        """Perform dataset validation.

        Args:
            df: Dataset to validate.
            schema: Expected schema for the dataset.
            **context: Additional validator-specific context.

        Returns:
            ValidationResult: Outcome of the validation step.
        """


class ValidationEngine:
    """Orchestrates all validators across ETAIQ raw datasets."""

    def __init__(self, validators: list[BaseValidator] | None = None) -> None:
        """Initialize the engine with optional custom validators.

        Args:
            validators: Validator instances; defaults to the full standard set.
        """
        if validators is None:
            from ml.validation.duplicate_validator import DuplicateValidator
            from ml.validation.foreign_key_validator import ForeignKeyValidator
            from ml.validation.gps_validator import GpsValidator
            from ml.validation.null_validator import NullValidator
            from ml.validation.schema_validator import SchemaValidator
            from ml.validation.target_validator import TargetValidator
            from ml.validation.timestamp_validator import TimestampValidator

            self._validators = [
                SchemaValidator(),
                NullValidator(),
                DuplicateValidator(),
                GpsValidator(),
                ForeignKeyValidator(),
                TimestampValidator(),
                TargetValidator(),
            ]
        else:
            self._validators = validators

    def run(
        self,
        datasets: dict[str, pd.DataFrame],
        schemas: dict[str, DatasetSchema],
    ) -> ValidationSummary:
        """Run all validators on the provided datasets.

        Args:
            datasets: Mapping of dataset name to DataFrame.
            schemas: Mapping of dataset name to schema definition.

        Returns:
            ValidationSummary: Aggregated validation outcomes and quality score.
        """
        summary = ValidationSummary()
        reference_ids: dict[str, set[str]] = {}

        for dataset_name, schema in schemas.items():
            df = datasets.get(dataset_name)
            if df is None:
                logger.warning("dataset_missing", dataset=dataset_name)
                continue

            if schema.id_column in df.columns:
                reference_ids[dataset_name] = (
                    df[schema.id_column].dropna().astype(str).unique().tolist()
                )
                reference_ids[dataset_name] = set(reference_ids[dataset_name])

            for validator in self._validators:
                context: dict[str, Any] = {"reference_ids": reference_ids}
                if dataset_name == ORDERS_SCHEMA.name:
                    context["orders_schema"] = ORDERS_SCHEMA

                if not self._applies(validator, schema):
                    continue

                result = validator.run(df, schema, **context)
                summary.results.append(result)

        calculator = QualityScoreCalculator()
        summary.quality_score, summary.component_scores = calculator.calculate(
            summary.results
        )
        return summary

    @staticmethod
    def _applies(validator: BaseValidator, schema: DatasetSchema) -> bool:
        """Determine whether a validator applies to a dataset schema.

        Args:
            validator: Validator instance to check.
            schema: Dataset schema definition.

        Returns:
            bool: True if the validator should run on this dataset.
        """
        if validator.name == "gps":
            return bool(schema.latitude_columns or schema.longitude_columns)
        if validator.name == "timestamp":
            return bool(schema.timestamp_columns)
        if validator.name == "target":
            return bool(schema.target_columns)
        if validator.name == "foreign_key":
            return schema.name == ORDERS_SCHEMA.name
        return True
