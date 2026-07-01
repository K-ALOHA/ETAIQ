"""Intelligent datetime detection without unnecessary parsing."""

from __future__ import annotations

import warnings

import pandas as pd

from ml.intelligence.config import DEFAULT_CONFIG, IntelligenceConfig


def column_name_suggests_datetime(
    column_name: str, config: IntelligenceConfig = DEFAULT_CONFIG
) -> bool:
    """Return True when a column name suggests temporal data.

    Args:
        column_name: Column header name.
        config: Intelligence configuration.

    Returns:
        bool: Whether the name matches datetime heuristics.
    """
    return bool(config.datetime_name_pattern.search(column_name))


def sample_values_match_datetime(
    series: pd.Series,
    config: IntelligenceConfig = DEFAULT_CONFIG,
) -> bool:
    """Check whether sample values match known datetime string patterns.

    Args:
        series: Column data.
        config: Intelligence configuration.

    Returns:
        bool: True when enough samples match datetime patterns.
    """
    sample = series.dropna().astype(str).str.strip().head(config.datetime_sample_size)
    if sample.empty:
        return False

    matches = 0
    for value in sample:
        if any(pattern.match(value) for pattern in config.datetime_value_patterns):
            matches += 1
    return (matches / len(sample)) >= config.datetime_match_ratio


def should_attempt_datetime_parse(
    column_name: str,
    series: pd.Series,
    config: IntelligenceConfig = DEFAULT_CONFIG,
) -> bool:
    """Decide whether datetime parsing should be attempted for a column.

    Args:
        column_name: Column header name.
        series: Column data.
        config: Intelligence configuration.

    Returns:
        bool: True only when name or sample values indicate datetime content.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    if column_name_suggests_datetime(column_name, config):
        return True
    return sample_values_match_datetime(series, config)


def infer_is_datetime(
    column_name: str,
    series: pd.Series,
    config: IntelligenceConfig = DEFAULT_CONFIG,
) -> bool:
    """Infer whether a column contains datetime values.

    Args:
        column_name: Column header name.
        series: Column data.
        config: Intelligence configuration.

    Returns:
        bool: True when the column is datetime-like.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    if not should_attempt_datetime_parse(column_name, series, config):
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(
            series.dropna().head(config.datetime_sample_size),
            errors="coerce",
            utc=True,
        )
    if parsed.empty:
        return False
    return float(parsed.notna().mean()) >= config.datetime_match_ratio
