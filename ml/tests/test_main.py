"""Integration tests for the validation CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.validation.main import run_validation


def test_run_validation_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()

    restaurants = pd.DataFrame(
        {
            "id": [1],
            "name": ["Test Kitchen"],
            "lat": [12.97],
            "lon": [77.59],
            "cuisine": ["Italian"],
            "avg_rating": [4.5],
            "prep_capacity": [15],
            "manager_contact": ["+91-9999999999"],
        }
    )
    riders = pd.DataFrame(
        {
            "id": [1],
            "lat": [12.98],
            "lon": [77.60],
            "vehicle_type": ["bike"],
            "completed_orders": [10],
            "shift_hours": [6.0],
            "current_load": [1],
            "rider_call_sign": ["R-001"],
        }
    )
    orders = pd.DataFrame(
        {
            "id": [1],
            "restaurant_id": [1],
            "rider_id": [1],
            "drop_lat": [12.99],
            "drop_lon": [77.58],
            "order_size": [2],
            "order_value": [25.0],
            "timestamp": ["2026-01-01T10:00:00Z"],
            "promised_eta": [15],
            "actual_delivery_time_min": [25.0],
            "order_status": ["delivered"],
            "promo_code_used": ["BLR10"],
        }
    )

    restaurants.to_csv(data_dir / "restaurants.csv", index=False)
    riders.to_csv(data_dir / "riders.csv", index=False)
    orders.to_csv(data_dir / "orders.csv", index=False)

    exit_code = run_validation(data_dir, reports_dir)
    assert exit_code == 0
    assert (reports_dir / "validation_report.json").exists()
    assert (reports_dir / "validation_report.md").exists()
    assert (reports_dir / "quality_score.json").exists()
