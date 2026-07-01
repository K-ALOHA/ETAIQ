"""Pytest configuration for ML package tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1].parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def restaurants_df() -> pd.DataFrame:
    """Valid restaurants dataset."""
    return pd.DataFrame(
        {
            "restaurant_id": ["R001", "R002"],
            "restaurant_name": ["Pizza Place", "Burger Hub"],
            "latitude": [12.97, 13.01],
            "longitude": [77.59, 77.61],
            "city": ["Bangalore", "Bangalore"],
            "is_active": [True, True],
        }
    )


@pytest.fixture
def riders_df() -> pd.DataFrame:
    """Valid riders dataset."""
    return pd.DataFrame(
        {
            "rider_id": ["D001", "D002"],
            "rider_name": ["Alex", "Sam"],
            "latitude": [12.98, 13.00],
            "longitude": [77.60, 77.62],
            "vehicle_type": ["bike", "bike"],
            "status": ["active", "active"],
        }
    )


@pytest.fixture
def orders_df() -> pd.DataFrame:
    """Valid orders dataset."""
    return pd.DataFrame(
        {
            "order_id": ["O001", "O002"],
            "restaurant_id": ["R001", "R002"],
            "rider_id": ["D001", "D002"],
            "order_timestamp": ["2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z"],
            "pickup_timestamp": ["2026-01-01T10:10:00Z", "2026-01-01T11:05:00Z"],
            "delivery_timestamp": ["2026-01-01T10:35:00Z", "2026-01-01T11:30:00Z"],
            "delivery_time_minutes": [35.0, 30.0],
            "customer_latitude": [12.99, 13.02],
            "customer_longitude": [77.58, 77.63],
        }
    )


@pytest.fixture
def reference_ids(
    restaurants_df: pd.DataFrame,
    riders_df: pd.DataFrame,
) -> dict[str, set[str]]:
    """Reference ID sets for foreign key validation."""
    return {
        "restaurants": set(restaurants_df["restaurant_id"].astype(str)),
        "riders": set(riders_df["rider_id"].astype(str)),
    }


@pytest.fixture
def schemas() -> dict:
    """All dataset schemas keyed by name."""
    from ml.validation.schemas import ORDERS_SCHEMA, RESTAURANTS_SCHEMA, RIDERS_SCHEMA

    return {
        RESTAURANTS_SCHEMA.name: RESTAURANTS_SCHEMA,
        RIDERS_SCHEMA.name: RIDERS_SCHEMA,
        ORDERS_SCHEMA.name: ORDERS_SCHEMA,
    }
