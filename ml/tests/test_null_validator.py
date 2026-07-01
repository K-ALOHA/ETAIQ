"""Tests for null validation."""

from __future__ import annotations

import pandas as pd

from ml.validation.null_validator import NullValidator
from ml.validation.schemas import RESTAURANTS_SCHEMA


def test_null_validator_passes_clean_data(restaurants_df: pd.DataFrame) -> None:
    result = NullValidator().validate(restaurants_df, RESTAURANTS_SCHEMA)
    assert result.passed is True
    assert result.details["non_nullable_violations"] == []


def test_null_validator_reports_nullable_columns(restaurants_df: pd.DataFrame) -> None:
    df = restaurants_df.copy()
    df.loc[0, "city"] = None
    result = NullValidator().validate(df, RESTAURANTS_SCHEMA)
    assert result.passed is True
    assert result.details["per_column"]["city"]["count"] == 1


def test_null_validator_fails_on_required_null(restaurants_df: pd.DataFrame) -> None:
    df = restaurants_df.copy()
    df.loc[0, "restaurant_id"] = None
    result = NullValidator().validate(df, RESTAURANTS_SCHEMA)
    assert result.passed is False
    assert "restaurant_id" in result.details["non_nullable_violations"]
