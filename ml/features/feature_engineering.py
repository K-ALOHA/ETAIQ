"""Architecture-only feature engineering engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureEngineeringConfig
from .feature_registry import FeatureRegistryManager
from .logging_config import FeatureEngineeringLogger


class FeatureEngineeringEngine:
    """Engine for feature engineering implementations."""

    EARTH_RADIUS_KM = 6371.0

    def __init__(
        self,
        config: FeatureEngineeringConfig | None = None,
        logger: FeatureEngineeringLogger | None = None,
        registry: FeatureRegistryManager | None = None,
    ) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.logger = logger or FeatureEngineeringLogger()
        self.registry = registry

    def create_temporal_features(self, training_df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from the detected timestamp column."""
        if self.registry is None:
            raise ValueError("Feature registry is required to detect the timestamp column.")

        timestamp_columns = [
            feature.name
            for feature in self.registry.list_features()
            if feature.feature_type == "Datetime"
        ]
        if not timestamp_columns:
            raise ValueError("No timestamp column detected in feature registry.")

        timestamp_column = self._select_timestamp_column(timestamp_columns)
        self.logger.info("Timestamp column detected", column=timestamp_column)

        dataframe = training_df.copy()
        dataframe[timestamp_column] = pd.to_datetime(dataframe[timestamp_column], errors="coerce")
        time_index = dataframe[timestamp_column]

        dataframe["hour"] = time_index.dt.hour
        dataframe["minute"] = time_index.dt.minute
        dataframe["day"] = time_index.dt.day
        dataframe["month"] = time_index.dt.month
        dataframe["weekday"] = time_index.dt.day_name()
        dataframe["is_weekend"] = dataframe["weekday"].isin(["Saturday", "Sunday"])

        meal_period = pd.Series("Night", index=dataframe.index)
        meal_period.loc[(dataframe["hour"] >= 5) & (dataframe["hour"] <= 10)] = "Breakfast"
        meal_period.loc[(dataframe["hour"] >= 11) & (dataframe["hour"] <= 15)] = "Lunch"
        meal_period.loc[(dataframe["hour"] >= 16) & (dataframe["hour"] <= 20)] = "Evening"
        dataframe["meal_period"] = meal_period

        dataframe["peak_hour"] = (
            ((dataframe["hour"] >= 7) & (dataframe["hour"] <= 10))
            | ((dataframe["hour"] >= 17) & (dataframe["hour"] <= 20))
        )

        new_features = [
            "hour",
            "minute",
            "day",
            "month",
            "weekday",
            "is_weekend",
            "meal_period",
            "peak_hour",
        ]

        self.logger.info("Temporal features created", features=len(new_features))

        print("=" * 40)
        print("Temporal Feature Summary")
        print("=" * 40)
        print(f"Timestamp Column: {timestamp_column}")
        print(f"New Features Created: {len(new_features)}")
        print(f"Current Dataset Shape: {dataframe.shape}")
        print("=" * 40)

        return dataframe

    def _select_timestamp_column(self, candidates: list[str]) -> str:
        if len(candidates) == 1:
            return candidates[0]
        for name in candidates:
            if name.lower() == "timestamp":
                return name
        return candidates[0]

    def create_geographical_features(self, training_df: pd.DataFrame) -> pd.DataFrame:
        """Create geographical features based on GPS coordinates."""
        if self.registry is None:
            raise ValueError("Feature registry is required to detect GPS columns.")

        gps_columns = [
            feature.name
            for feature in self.registry.list_features()
            if feature.feature_type == "GPS"
        ]
        if not gps_columns:
            raise ValueError("No GPS columns detected in feature registry.")

        restaurant_lat, restaurant_lon, drop_lat, drop_lon = self._resolve_gps_columns(gps_columns)
        self.logger.info(
            "GPS columns detected",
            restaurant_lat=restaurant_lat,
            restaurant_lon=restaurant_lon,
            drop_lat=drop_lat,
            drop_lon=drop_lon,
        )

        dataframe = training_df.copy()
        dataframe["distance_km"] = self.calculate_haversine_distance(
            dataframe[restaurant_lat],
            dataframe[restaurant_lon],
            dataframe[drop_lat],
            dataframe[drop_lon],
        )
        dataframe["latitude_difference"] = (dataframe[restaurant_lat] - dataframe[drop_lat]).abs()
        dataframe["longitude_difference"] = (dataframe[restaurant_lon] - dataframe[drop_lon]).abs()
        dataframe["same_location"] = dataframe["distance_km"] < 0.05

        new_features = [
            "distance_km",
            "latitude_difference",
            "longitude_difference",
            "same_location",
        ]

        self.logger.info("Haversine calculation completed")
        self.logger.info("Geographical features created", features=len(new_features))

        print("=" * 40)
        print("Geographical Feature Summary")
        print("=" * 40)
        print(f"GPS Columns Detected: {restaurant_lat}, {restaurant_lon}, {drop_lat}, {drop_lon}")
        print("Distance Feature Created")
        print("Latitude Difference Created")
        print("Longitude Difference Created")
        print("Same Location Created")
        print(f"Current Dataset Shape: {dataframe.shape}")
        print("=" * 40)

        return dataframe

    def calculate_haversine_distance(
        self,
        restaurant_lat: pd.Series,
        restaurant_lon: pd.Series,
        drop_lat: pd.Series,
        drop_lon: pd.Series,
    ) -> pd.Series:
        """Calculate great-circle distance between pairs of GPS coordinates."""
        lat1 = np.radians(restaurant_lat.astype(float))
        lon1 = np.radians(restaurant_lon.astype(float))
        lat2 = np.radians(drop_lat.astype(float))
        lon2 = np.radians(drop_lon.astype(float))

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return pd.Series(self.EARTH_RADIUS_KM * c, index=restaurant_lat.index)

    def create_operational_features(self, training_df: pd.DataFrame) -> pd.DataFrame:
        """Create operational features from the merged training dataframe."""
        if self.registry is None:
            raise ValueError("Feature registry is required to detect operational columns.")

        registry_columns = {feature.name for feature in self.registry.list_features()}
        required_columns = [
            "prep_capacity",
            "current_load",
            "completed_orders",
            "shift_hours",
            "order_size",
            "order_value",
        ]
        missing = [column for column in required_columns if column not in registry_columns]
        if missing:
            raise ValueError(f"Missing required operational columns: {', '.join(missing)}")

        self.logger.info("Operational columns detected", columns=required_columns)

        dataframe = training_df.copy()
        prep_capacity = dataframe["prep_capacity"].astype(float)
        current_load = dataframe["current_load"].astype(float)
        completed_orders = dataframe["completed_orders"].astype(float)
        shift_hours = dataframe["shift_hours"].astype(float)
        order_size = dataframe["order_size"].astype(float)
        order_value = dataframe["order_value"].astype(float)

        dataframe["load_ratio"] = self._safe_divide(current_load, prep_capacity)
        dataframe["remaining_capacity"] = prep_capacity - current_load
        dataframe["capacity_utilization_percent"] = self._safe_divide(current_load, prep_capacity) * 100
        dataframe["orders_per_shift_hour"] = self._safe_divide(completed_orders, shift_hours)
        dataframe["average_order_value_per_item"] = self._safe_divide(order_value, order_size)
        dataframe["high_workload"] = dataframe["load_ratio"] >= 0.80

        self.logger.info("Operational features created", features=6)

        print("=" * 40)
        print("Operational Feature Summary")
        print("=" * 40)
        print("Operational Columns Detected")
        print("Load Ratio Created")
        print("Remaining Capacity Created")
        print("Capacity Utilization Created")
        print("Orders Per Shift Hour Created")
        print("Average Order Value Per Item Created")
        print("High Workload Created")
        print(f"Current Dataset Shape: {dataframe.shape}")
        print("=" * 40)

        return dataframe

    def create_business_features(self, training_df: pd.DataFrame) -> pd.DataFrame:
        """Create business-oriented features from the merged training dataframe."""
        if self.registry is None:
            raise ValueError("Feature registry is required to detect business columns.")

        if self.registry is None:
            raise ValueError("Feature registry is required to detect business columns.")

        dataframe_columns = set(training_df.columns)
        required_columns = [
            "completed_orders",
            "avg_rating",
            "order_value",
            "order_size",
            "load_ratio",
            "prep_capacity",
        ]
        missing_in_df = [column for column in required_columns if column not in dataframe_columns]
        if missing_in_df:
            raise ValueError(f"Missing required business columns: {', '.join(missing_in_df)}")

        self.logger.info("Business columns detected", columns=required_columns)

        dataframe = training_df.copy()
        completed_orders = dataframe["completed_orders"].astype(float)
        avg_rating = dataframe["avg_rating"].astype(float)
        order_value = dataframe["order_value"].astype(float)
        order_size = dataframe["order_size"].astype(float)
        load_ratio = dataframe["load_ratio"].astype(float)
        prep_capacity = dataframe["prep_capacity"].astype(float)

        experience_conditions = [
            completed_orders < 500,
            completed_orders.between(500, 1499),
            completed_orders.between(1500, 2999),
            completed_orders >= 3000,
        ]
        experience_choices = ["Beginner", "Intermediate", "Experienced", "Expert"]
        dataframe["rider_experience_level"] = np.select(experience_conditions, experience_choices, default="Beginner")

        quality_conditions = [
            avg_rating < 3.5,
            avg_rating.between(3.5, 4.2),
            avg_rating.between(4.2, 4.7),
            avg_rating >= 4.7,
        ]
        quality_choices = ["Low", "Standard", "Good", "Premium"]
        dataframe["restaurant_quality_tier"] = np.select(quality_conditions, quality_choices, default="Low")

        high_value_threshold = np.nanpercentile(order_value.dropna(), 75)
        large_order_threshold = np.nanpercentile(order_size.dropna(), 75)

        dataframe["high_value_order"] = order_value >= high_value_threshold
        dataframe["large_order"] = order_size >= large_order_threshold
        dataframe["premium_restaurant"] = avg_rating >= 4.5

        prep_capacity_median = np.nanmedian(prep_capacity)
        dataframe["busy_restaurant"] = (load_ratio >= 0.80) & (prep_capacity > prep_capacity_median)

        new_features = [
            "rider_experience_level",
            "restaurant_quality_tier",
            "high_value_order",
            "large_order",
            "premium_restaurant",
            "busy_restaurant",
        ]

        self.logger.info("Business features created", features=len(new_features))

        print("=" * 40)
        print("Business Feature Summary")
        print("=" * 40)
        print("Rider Experience Created")
        print("Restaurant Quality Created")
        print("High Value Order Created")
        print("Large Order Created")
        print("Premium Restaurant Created")
        print("Busy Restaurant Created")
        print(f"Current Dataset Shape: {dataframe.shape}")
        print("=" * 40)

        return dataframe

    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Divide two series safely, replacing inf and NaN with zero."""
        result = numerator / denominator
        result = result.replace([np.inf, -np.inf], 0.0)
        return result.fillna(0.0)

    def _resolve_gps_columns(self, gps_columns: list[str]) -> tuple[str, str, str, str]:
        lower_names = {name.lower(): name for name in gps_columns}

        drop_lat = self._match_column(lower_names, ["drop_lat", "droplatitude", "drop_latitude"], ["_rider"])
        drop_lon = self._match_column(lower_names, ["drop_lon", "droplongitude", "drop_longitude"], ["_rider"])

        restaurant_candidates = [name for name in gps_columns if "drop" not in name.lower() and "rider" not in name.lower()]
        restaurant_lat = self._match_column({name.lower(): name for name in restaurant_candidates}, ["restaurant_lat", "restaurantlatitude", "lat"], [])
        restaurant_lon = self._match_column({name.lower(): name for name in restaurant_candidates}, ["restaurant_lon", "restaurantlongitude", "lon"], [])

        missing = [
            column_name
            for column_name, value in (
                ("restaurant_lat", restaurant_lat),
                ("restaurant_lon", restaurant_lon),
                ("drop_lat", drop_lat),
                ("drop_lon", drop_lon),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"Missing required GPS columns: {', '.join(missing)}")

        return restaurant_lat, restaurant_lon, drop_lat, drop_lon

    def _match_column(
        self,
        allowed_columns: dict[str, str],
        preferred_keys: list[str],
        exclude_substrings: list[str],
    ) -> str | None:
        for preferred in preferred_keys:
            if preferred in allowed_columns and all(exclude not in preferred for exclude in exclude_substrings):
                return allowed_columns[preferred]

        candidates = [
            name
            for lower, name in allowed_columns.items()
            if any(token in lower for token in preferred_keys)
            and not any(exclude in lower for exclude in exclude_substrings)
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def run(self) -> None:
        """Compatibility stub for feature engineering initialization."""
        pass
