"""Tests for foreign key validation."""

from __future__ import annotations

import pandas as pd

from ml.validation.foreign_key_validator import ForeignKeyValidator
from ml.validation.schemas import ORDERS_SCHEMA


def test_foreign_key_validator_passes_valid_references(
    orders_df: pd.DataFrame,
    reference_ids: dict[str, set[str]],
) -> None:
    result = ForeignKeyValidator().validate(
        orders_df,
        ORDERS_SCHEMA,
        reference_ids=reference_ids,
    )
    assert result.passed is True
    assert result.details["total_orphans"] == 0


def test_foreign_key_validator_detects_orphan_restaurant(
    orders_df: pd.DataFrame,
    reference_ids: dict[str, set[str]],
) -> None:
    df = orders_df.copy()
    df.loc[0, "restaurant_id"] = 999
    result = ForeignKeyValidator().validate(
        df, ORDERS_SCHEMA, reference_ids=reference_ids
    )
    assert result.passed is False
    assert result.details["per_column"]["restaurant_id"]["orphan_count"] == 1


def test_foreign_key_validator_detects_orphan_rider(
    orders_df: pd.DataFrame,
    reference_ids: dict[str, set[str]],
) -> None:
    df = orders_df.copy()
    df.loc[0, "rider_id"] = 999
    result = ForeignKeyValidator().validate(
        df, ORDERS_SCHEMA, reference_ids=reference_ids
    )
    assert result.passed is False
    assert result.details["per_column"]["rider_id"]["orphan_count"] == 1
