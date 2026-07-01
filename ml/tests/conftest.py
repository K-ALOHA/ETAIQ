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
            "id": [1, 2],
            "name": ["Pizza Place", "Burger Hub"],
            "lat": [12.97, 13.01],
            "lon": [77.59, 77.61],
            "cuisine": ["Italian", "Indian"],
            "avg_rating": [4.5, 4.2],
            "prep_capacity": [15, 20],
            "manager_contact": ["+91-9999999999", "+91-8888888888"],
        }
    )


@pytest.fixture
def riders_df() -> pd.DataFrame:
    """Valid riders dataset."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "lat": [12.98, 13.00],
            "lon": [77.60, 77.62],
            "vehicle_type": ["bike", "bike"],
            "completed_orders": [10, 20],
            "shift_hours": [6.0, 6.5],
            "current_load": [1, 0],
            "rider_call_sign": ["R-001", "R-002"],
        }
    )


@pytest.fixture
def orders_df() -> pd.DataFrame:
    """Valid orders dataset."""
    return pd.DataFrame(
        {
            "id": [1, 2],
            "restaurant_id": [1, 2],
            "rider_id": [1, 2],
            "drop_lat": [12.99, 13.02],
            "drop_lon": [77.58, 77.63],
            "order_size": [2, 1],
            "order_value": [35.0, 30.0],
            "timestamp": ["2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z"],
            "promised_eta": [15, 20],
            "actual_delivery_time_min": [35.0, 30.0],
            "order_status": ["delivered", "delivered"],
            "promo_code_used": ["BLR10", "WELCOME50"],
        }
    )


@pytest.fixture
def reference_ids(
    restaurants_df: pd.DataFrame,
    riders_df: pd.DataFrame,
) -> dict[str, set[str]]:
    """Reference ID sets for foreign key validation."""
    return {
        "restaurants": set(restaurants_df["id"].astype(str)),
        "riders": set(riders_df["id"].astype(str)),
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
