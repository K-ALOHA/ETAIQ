"""Tests for target variable validation."""

from __future__ import annotations

import pandas as pd

from ml.validation.schemas import ORDERS_SCHEMA
from ml.validation.target_validator import TargetValidator


def test_target_validator_passes_valid_delivery_times(orders_df: pd.DataFrame) -> None:
    result = TargetValidator().validate(orders_df, ORDERS_SCHEMA)
    assert result.passed is True
    assert result.details["total_invalid"] == 0


def test_target_validator_detects_negative_delivery_time(
    orders_df: pd.DataFrame,
) -> None:
    df = orders_df.copy()
    df.loc[0, "delivery_time_minutes"] = -5.0
    result = TargetValidator().validate(df, ORDERS_SCHEMA)
    assert result.passed is False
    assert result.details["per_column"]["delivery_time_minutes"]["negative_count"] == 1


def test_target_validator_detects_impossible_delivery_time(
    orders_df: pd.DataFrame,
) -> None:
    df = orders_df.copy()
    df.loc[0, "delivery_time_minutes"] = 5000.0
    result = TargetValidator().validate(df, ORDERS_SCHEMA)
    assert result.passed is False
    assert (
        result.details["per_column"]["delivery_time_minutes"]["impossible_high_count"]
        == 1
    )
