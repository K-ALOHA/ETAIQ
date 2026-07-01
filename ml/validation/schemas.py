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
    id_column="id",
    latitude_columns=("lat",),
    longitude_columns=("lon",),
    columns=(
        ColumnSpec("id", ColumnDtype.INTEGER),
        ColumnSpec("name", ColumnDtype.STRING),
        ColumnSpec("lat", ColumnDtype.FLOAT),
        ColumnSpec("lon", ColumnDtype.FLOAT),
        ColumnSpec("cuisine", ColumnDtype.STRING, nullable=True),
        ColumnSpec("avg_rating", ColumnDtype.FLOAT, nullable=True),
        ColumnSpec("prep_capacity", ColumnDtype.INTEGER, nullable=True),
        ColumnSpec("manager_contact", ColumnDtype.STRING, nullable=True),
    ),
)

RIDERS_SCHEMA = DatasetSchema(
    name="riders",
    filename="riders.csv",
    id_column="id",
    latitude_columns=("lat",),
    longitude_columns=("lon",),
    columns=(
        ColumnSpec("id", ColumnDtype.INTEGER),
        ColumnSpec("lat", ColumnDtype.FLOAT),
        ColumnSpec("lon", ColumnDtype.FLOAT),
        ColumnSpec("vehicle_type", ColumnDtype.STRING, nullable=True),
        ColumnSpec("completed_orders", ColumnDtype.INTEGER, nullable=True),
        ColumnSpec("shift_hours", ColumnDtype.FLOAT, nullable=True),
        ColumnSpec("current_load", ColumnDtype.INTEGER, nullable=True),
        ColumnSpec("rider_call_sign", ColumnDtype.STRING, nullable=True),
    ),
)

ORDERS_SCHEMA = DatasetSchema(
    name="orders",
    filename="orders.csv",
    id_column="id",
    latitude_columns=("drop_lat",),
    longitude_columns=("drop_lon",),
    timestamp_columns=("timestamp",),
    target_columns=("actual_delivery_time_min",),
    columns=(
        ColumnSpec("id", ColumnDtype.INTEGER),
        ColumnSpec("restaurant_id", ColumnDtype.INTEGER),
        ColumnSpec("rider_id", ColumnDtype.INTEGER),
        ColumnSpec("drop_lat", ColumnDtype.FLOAT, nullable=True),
        ColumnSpec("drop_lon", ColumnDtype.FLOAT, nullable=True),
        ColumnSpec("order_size", ColumnDtype.INTEGER, nullable=True),
        ColumnSpec("order_value", ColumnDtype.FLOAT, nullable=True),
        ColumnSpec("timestamp", ColumnDtype.DATETIME),
        ColumnSpec("promised_eta", ColumnDtype.INTEGER, nullable=True),
        ColumnSpec("actual_delivery_time_min", ColumnDtype.FLOAT, nullable=True),
        ColumnSpec("order_status", ColumnDtype.STRING, nullable=True),
        ColumnSpec("promo_code_used", ColumnDtype.STRING, nullable=True),
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
