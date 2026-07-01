"""Tests for GPS validation."""

from __future__ import annotations

import pandas as pd

from ml.validation.gps_validator import GpsValidator
from ml.validation.schemas import RESTAURANTS_SCHEMA


def test_gps_validator_passes_valid_coordinates(restaurants_df: pd.DataFrame) -> None:
    result = GpsValidator().validate(restaurants_df, RESTAURANTS_SCHEMA)
    assert result.passed is True
    assert sum(result.details["invalid_counts"].values()) == 0


def test_gps_validator_detects_invalid_latitude(restaurants_df: pd.DataFrame) -> None:
    df = restaurants_df.copy()
    df.loc[0, "latitude"] = 95.0
    result = GpsValidator().validate(df, RESTAURANTS_SCHEMA)
    assert result.passed is False
    assert result.details["invalid_counts"]["latitude"] == 1


def test_gps_validator_detects_invalid_longitude(restaurants_df: pd.DataFrame) -> None:
    df = restaurants_df.copy()
    df.loc[0, "longitude"] = 200.0
    result = GpsValidator().validate(df, RESTAURANTS_SCHEMA)
    assert result.passed is False
    assert result.details["invalid_counts"]["longitude"] == 1
