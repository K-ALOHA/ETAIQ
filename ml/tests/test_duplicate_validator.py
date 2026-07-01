"""Tests for duplicate validation."""

from __future__ import annotations

import pandas as pd

from ml.validation.duplicate_validator import DuplicateValidator
from ml.validation.schemas import ORDERS_SCHEMA


def test_duplicate_validator_passes_unique_data(orders_df: pd.DataFrame) -> None:
    result = DuplicateValidator().validate(orders_df, ORDERS_SCHEMA)
    assert result.passed is True
    assert result.details["duplicate_id_count"] == 0
    assert result.details["exact_duplicate_rows"] == 0


def test_duplicate_validator_detects_duplicate_ids(orders_df: pd.DataFrame) -> None:
    df = pd.concat([orders_df, orders_df.iloc[[0]]], ignore_index=True)
    result = DuplicateValidator().validate(df, ORDERS_SCHEMA)
    assert result.passed is False
    assert result.details["duplicate_id_count"] == 2


def test_duplicate_validator_detects_exact_duplicate_rows(
    orders_df: pd.DataFrame,
) -> None:
    df = pd.concat([orders_df, orders_df.iloc[[0]]], ignore_index=True)
    result = DuplicateValidator().validate(df, ORDERS_SCHEMA)
    assert result.details["exact_duplicate_rows"] >= 1
