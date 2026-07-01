"""Dataset schema definitions for ETAIQ raw CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ColumnDtype(str, Enum):
    """Supported logical column data types."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single dataset column.

    Attributes:
        name: Column name as it appears in the CSV header.
        dtype: Expected logical data type.
        required: Whether the column must be present.
        nullable: Whether null values are permitted.
    """

    name: str
    dtype: ColumnDtype
    required: bool = True
    nullable: bool = False


@dataclass(frozen=True)
class DatasetSchema:
    """Complete schema for a raw ETAIQ dataset.

    Attributes:
        name: Dataset identifier used in logs and reports.
        filename: Expected CSV filename under ``ml/data/raw/``.
        columns: Ordered column specifications.
        id_column: Primary identifier column for duplicate checks.
        latitude_columns: Columns validated as geographic latitude.
        longitude_columns: Columns validated as geographic longitude.
        timestamp_columns: Columns validated as parseable timestamps.
        target_columns: Regression target columns (e.g. delivery time).
    """

    name: str
    filename: str
    columns: tuple[ColumnSpec, ...]
    id_column: str
    latitude_columns: tuple[str, ...] = ()
    longitude_columns: tuple[str, ...] = ()
    timestamp_columns: tuple[str, ...] = ()
    target_columns: tuple[str, ...] = ()


RESTAURANTS_SCHEMA = DatasetSchema(
    name="restaurants",
    filename="restaurants.csv",
    id_column="restaurant_id",
    latitude_columns=("latitude",),
    longitude_columns=("longitude",),
    columns=(
        ColumnSpec("restaurant_id", ColumnDtype.STRING),
        ColumnSpec("restaurant_name", ColumnDtype.STRING),
        ColumnSpec("latitude", ColumnDtype.FLOAT),
        ColumnSpec("longitude", ColumnDtype.FLOAT),
        ColumnSpec("city", ColumnDtype.STRING, nullable=True),
        ColumnSpec("is_active", ColumnDtype.BOOLEAN, nullable=True),
    ),
)

RIDERS_SCHEMA = DatasetSchema(
    name="riders",
    filename="riders.csv",
    id_column="rider_id",
    latitude_columns=("latitude",),
    longitude_columns=("longitude",),
    columns=(
        ColumnSpec("rider_id", ColumnDtype.STRING),
        ColumnSpec("rider_name", ColumnDtype.STRING),
        ColumnSpec("latitude", ColumnDtype.FLOAT),
        ColumnSpec("longitude", ColumnDtype.FLOAT),
        ColumnSpec("vehicle_type", ColumnDtype.STRING, nullable=True),
        ColumnSpec("status", ColumnDtype.STRING, nullable=True),
    ),
)

ORDERS_SCHEMA = DatasetSchema(
    name="orders",
    filename="orders.csv",
    id_column="order_id",
    latitude_columns=("customer_latitude",),
    longitude_columns=("customer_longitude",),
    timestamp_columns=("order_timestamp", "pickup_timestamp", "delivery_timestamp"),
    target_columns=("delivery_time_minutes",),
    columns=(
        ColumnSpec("order_id", ColumnDtype.STRING),
        ColumnSpec("restaurant_id", ColumnDtype.STRING),
        ColumnSpec("rider_id", ColumnDtype.STRING),
        ColumnSpec("order_timestamp", ColumnDtype.DATETIME),
        ColumnSpec("pickup_timestamp", ColumnDtype.DATETIME, nullable=True),
        ColumnSpec("delivery_timestamp", ColumnDtype.DATETIME, nullable=True),
        ColumnSpec("delivery_time_minutes", ColumnDtype.FLOAT),
        ColumnSpec("customer_latitude", ColumnDtype.FLOAT, nullable=True),
        ColumnSpec("customer_longitude", ColumnDtype.FLOAT, nullable=True),
    ),
)

ALL_SCHEMAS: tuple[DatasetSchema, ...] = (
    RESTAURANTS_SCHEMA,
    RIDERS_SCHEMA,
    ORDERS_SCHEMA,
)

SCHEMA_BY_NAME: dict[str, DatasetSchema] = {
    schema.name: schema for schema in ALL_SCHEMAS
}
