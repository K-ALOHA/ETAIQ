"""Tests for timestamp validation."""

from __future__ import annotations

import pandas as pd

from ml.validation.timestamp_validator import TimestampValidator
from ml.validation.schemas import ORDERS_SCHEMA


def test_timestamp_validator_passes_valid_timestamps(orders_df: pd.DataFrame) -> None:
    result = TimestampValidator().validate(orders_df, ORDERS_SCHEMA)
    assert result.passed is True
    assert result.details["total_invalid"] == 0


def test_timestamp_validator_detects_invalid_values(orders_df: pd.DataFrame) -> None:
    df = orders_df.copy()
    df.loc[0, "order_timestamp"] = "not-a-date"
    result = TimestampValidator().validate(df, ORDERS_SCHEMA)
    assert result.passed is False
    assert result.details["per_column"]["order_timestamp"]["invalid_count"] == 1


def test_timestamp_validator_ignores_nullable_nulls(orders_df: pd.DataFrame) -> None:
    df = orders_df.copy()
    df.loc[0, "pickup_timestamp"] = None
    result = TimestampValidator().validate(df, ORDERS_SCHEMA)
    assert result.details["per_column"]["pickup_timestamp"]["invalid_count"] == 0
