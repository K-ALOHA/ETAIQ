"""Tests for schema validation."""

from __future__ import annotations

import pandas as pd

from ml.validation.schema_validator import SchemaValidator
from ml.validation.schemas import RESTAURANTS_SCHEMA


def test_schema_validator_passes_valid_dataframe(restaurants_df: pd.DataFrame) -> None:
    validator = SchemaValidator()
    result = validator.validate(restaurants_df, RESTAURANTS_SCHEMA)
    assert result.passed is True
    assert result.score == 100.0
    assert result.details["missing_columns"] == []


def test_schema_validator_detects_missing_columns(restaurants_df: pd.DataFrame) -> None:
    df = restaurants_df.drop(columns=["latitude", "longitude"])
    result = SchemaValidator().validate(df, RESTAURANTS_SCHEMA)
    assert result.passed is False
    assert "latitude" in result.details["missing_columns"]
    assert "longitude" in result.details["missing_columns"]


def test_schema_validator_detects_extra_columns(restaurants_df: pd.DataFrame) -> None:
    df = restaurants_df.copy()
    df["unexpected_col"] = "x"
    result = SchemaValidator().validate(df, RESTAURANTS_SCHEMA)
    assert "unexpected_col" in result.details["extra_columns"]


def test_schema_validator_detects_invalid_latitude_dtype(
    restaurants_df: pd.DataFrame,
) -> None:
    df = restaurants_df.copy()
    df["latitude"] = ["not-a-float", "also-bad"]
    result = SchemaValidator().validate(df, RESTAURANTS_SCHEMA)
    assert "latitude" in result.details["dtype_issues"]
