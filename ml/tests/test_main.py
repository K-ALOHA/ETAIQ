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
            "restaurant_id": ["R001"],
            "restaurant_name": ["Test Kitchen"],
            "latitude": [12.97],
            "longitude": [77.59],
            "city": ["Bangalore"],
            "is_active": [True],
        }
    )
    riders = pd.DataFrame(
        {
            "rider_id": ["D001"],
            "rider_name": ["Rider"],
            "latitude": [12.98],
            "longitude": [77.60],
            "vehicle_type": ["bike"],
            "status": ["active"],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": ["O001"],
            "restaurant_id": ["R001"],
            "rider_id": ["D001"],
            "order_timestamp": ["2026-01-01T10:00:00Z"],
            "pickup_timestamp": ["2026-01-01T10:10:00Z"],
            "delivery_timestamp": ["2026-01-01T10:35:00Z"],
            "delivery_time_minutes": [25.0],
            "customer_latitude": [12.99],
            "customer_longitude": [77.58],
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
