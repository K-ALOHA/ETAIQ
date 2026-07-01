"""Tests for datetime detection."""

from __future__ import annotations

import warnings

import pandas as pd

from ml.intelligence.datetime_detector import (
    column_name_suggests_datetime,
    infer_is_datetime,
    sample_values_match_datetime,
    should_attempt_datetime_parse,
)


def test_column_name_suggests_datetime() -> None:
    assert column_name_suggests_datetime("order_timestamp")
    assert not column_name_suggests_datetime("restaurant_name")


def test_sample_values_match_iso_datetime() -> None:
    series = pd.Series(["2026-01-01T10:00:00Z", "2026-01-02T11:00:00Z"])
    assert sample_values_match_datetime(series)


def test_does_not_parse_plain_strings() -> None:
    series = pd.Series(["Pizza Place", "Burger Hub", "Sushi Spot"])
    assert not should_attempt_datetime_parse("restaurant_name", series)


def test_infer_datetime_without_warnings() -> None:
    series = pd.Series(["2026-01-01T10:00:00Z", "2026-01-02T11:00:00Z"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = infer_is_datetime("created_at", series)
    assert result is True
    assert not any("Could not infer format" in str(w.message) for w in caught)
