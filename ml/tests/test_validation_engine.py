"""Integration tests for the validation engine."""

from __future__ import annotations

import pandas as pd

from ml.validation.validator import ValidationEngine


def test_validation_engine_runs_all_applicable_validators(
    restaurants_df: pd.DataFrame,
    riders_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    schemas: dict,
) -> None:
    datasets = {
        "restaurants": restaurants_df,
        "riders": riders_df,
        "orders": orders_df,
    }
    summary = ValidationEngine().run(datasets, schemas)
    assert summary.quality_score > 0
    validator_names = {r.validator_name for r in summary.results}
    assert "schema" in validator_names
    assert "foreign_key" in validator_names
    assert "target" in validator_names
    assert all(r.passed for r in summary.results)


def test_validation_engine_detects_issues(
    restaurants_df: pd.DataFrame,
    riders_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    schemas: dict,
) -> None:
    bad_orders = orders_df.copy()
    bad_orders.loc[0, "actual_delivery_time_min"] = -10.0
    bad_orders.loc[1, "restaurant_id"] = 999

    summary = ValidationEngine().run(
        {
            "restaurants": restaurants_df,
            "riders": riders_df,
            "orders": bad_orders,
        },
        schemas,
    )
    failed = [r for r in summary.results if not r.passed]
    assert len(failed) >= 2
