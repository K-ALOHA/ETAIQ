"""Shared fixtures for intelligence engine tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def restaurants_frame() -> pd.DataFrame:
    """Synthetic restaurants dataset."""
    return pd.DataFrame(
        {
            "restaurant_id": ["R001", "R002", "R003"],
            "restaurant_name": ["Pizza Place", "Burger Hub", "Sushi Spot"],
            "latitude": [12.97, 13.01, 12.95],
            "longitude": [77.59, 77.61, 77.57],
            "city": ["Bangalore", "Bangalore", "Bangalore"],
            "is_active": [True, True, False],
        }
    )


@pytest.fixture
def riders_frame() -> pd.DataFrame:
    """Synthetic riders dataset."""
    return pd.DataFrame(
        {
            "rider_id": ["D001", "D002", "D003"],
            "rider_name": ["Alex Rider", "Sam Courier", "Pat Dash"],
            "latitude": [12.98, 13.00, 12.96],
            "longitude": [77.60, 77.62, 77.58],
            "vehicle_type": ["bike", "bike", "scooter"],
            "status": ["active", "active", "inactive"],
        }
    )


@pytest.fixture
def orders_frame() -> pd.DataFrame:
    """Synthetic orders dataset with target and join keys."""
    return pd.DataFrame(
        {
            "order_id": ["O001", "O002", "O003"],
            "restaurant_id": ["R001", "R002", "R001"],
            "rider_id": ["D001", "D002", "D003"],
            "order_timestamp": [
                "2026-01-01T10:00:00Z",
                "2026-01-01T11:00:00Z",
                "2026-01-01T12:00:00Z",
            ],
            "delivery_timestamp": [
                "2026-01-01T10:35:00Z",
                "2026-01-01T11:30:00Z",
                "2026-01-01T12:40:00Z",
            ],
            "delivery_time_minutes": [35.0, 30.0, 40.0],
            "customer_latitude": [12.99, 13.02, 12.94],
            "customer_longitude": [77.58, 77.63, 77.56],
            "delivery_time_estimate": [34.5, 29.8, 39.7],
        }
    )


@pytest.fixture
def frames(
    restaurants_frame: pd.DataFrame,
    riders_frame: pd.DataFrame,
    orders_frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Named synthetic dataset frames."""
    return {
        "restaurants": restaurants_frame,
        "riders": riders_frame,
        "orders": orders_frame,
    }


@pytest.fixture
def raw_data_dir(
    tmp_path: Path,
    restaurants_frame: pd.DataFrame,
    riders_frame: pd.DataFrame,
    orders_frame: pd.DataFrame,
) -> Path:
    """Temporary raw data directory with nested CSV files."""
    nested = tmp_path / "nested"
    nested.mkdir()
    restaurants_frame.to_csv(tmp_path / "restaurants.csv", index=False)
    riders_frame.to_csv(nested / "riders.csv", index=False)
    orders_frame.to_csv(tmp_path / "orders.csv", index=False)
    return tmp_path
