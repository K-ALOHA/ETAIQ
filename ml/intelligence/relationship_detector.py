"""Automatic primary-key and foreign-key relationship detection."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from ml.intelligence.config import DEFAULT_CONFIG, IntelligenceConfig
from ml.intelligence.logging_config import get_logger
from ml.intelligence.models import ColumnProfile, DatasetProfile, Relationship

logger = get_logger(__name__)


class RelationshipDetector:
    """Detects join relationships using overlap, uniqueness, and referential integrity."""

    def __init__(self, config: IntelligenceConfig = DEFAULT_CONFIG) -> None:
        """Initialize the detector.

        Args:
            config: Intelligence configuration.
        """
        self._config = config

    def detect(
        self,
        frames: dict[str, pd.DataFrame],
        profiles: list[DatasetProfile],
    ) -> tuple[list[Relationship], dict[tuple[str, str], tuple[str, str]]]:
        """Detect relationships across all dataset pairs."""
        logger.info("relationship_detection_start", datasets=len(frames))
        start = time.perf_counter()

        profile_map: dict[tuple[str, str], ColumnProfile] = {}
        for p in profiles:
            for col in p.columns:
                profile_map[(p.dataset_id, col.name)] = col

        primary_keys = self._detect_primary_keys(profiles)
        relationships: list[Relationship] = []
        foreign_key_map: dict[tuple[str, str], tuple[str, str]] = {}
        dataset_ids = list(frames.keys())

        # Configurable weights
        semantic_weight = getattr(self._config, "semantic_weight", 0.40)
        referential_weight = getattr(self._config, "referential_weight", 0.30)
        cardinality_weight = getattr(self._config, "cardinality_weight", 0.15)
        datatype_weight = getattr(self._config, "datatype_weight", 0.10)
        overlap_weight = getattr(self._config, "overlap_weight", 0.05)

        for source_id in dataset_ids:
            source_frame = frames[source_id]
            for column in source_frame.columns:
                source_col = str(column)
                source_values = self._normalized_values(source_frame[column])
                if source_values.empty:
                    continue

                for target_id in dataset_ids:
                    if source_id == target_id:
                        continue
                    target_frame = frames[target_id]
                    for target_column in target_frame.columns:
                        target_col = str(target_column)
                        target_values = self._normalized_values(
                            target_frame[target_column]
                        )
                        if target_values.empty:
                            continue

                        # Stage 1: Candidate Discovery
                        if not self._is_candidate(
                            source_id, source_col, target_id, target_col, profile_map
                        ):
                            continue

                        # Stage 2: Evidence Scoring
                        semantic_sim = self._semantic_similarity(
                            source_id, source_col, target_id, target_col
                        )
                        referential = self._referential_integrity(
                            source_values, target_values
                        )

                        target_is_pk = (target_id, target_col) in primary_keys
                        cardinality = self._infer_cardinality(
                            source_values, target_values, target_is_pk
                        )
                        if cardinality in ("many_to_one", "one_to_one"):
                            cardinality_score = 1.0
                        elif cardinality == "one_to_many":
                            cardinality_score = 0.8
                        else:  # many_to_many
                            cardinality_score = 0.1

                        src_profile = profile_map[(source_id, source_col)]
                        tgt_profile = profile_map[(target_id, target_col)]
                        src_dtype = src_profile.inferred_dtype
                        tgt_dtype = tgt_profile.inferred_dtype
                        if src_dtype == tgt_dtype:
                            datatype_score = 1.0
                        elif src_dtype in ("integer", "float") and tgt_dtype in (
                            "integer",
                            "float",
                        ):
                            datatype_score = 0.8
                        else:
                            datatype_score = 0.0

                        overlap = self._overlap_ratio(source_values, target_values)

                        confidence = (
                            semantic_sim * semantic_weight
                            + referential * referential_weight
                            + cardinality_score * cardinality_weight
                            + datatype_score * datatype_weight
                            + overlap * overlap_weight
                        )

                        # Stage 3: Business Validation
                        is_valid_business = self._validate_business_sense(
                            source_id, source_col, target_id, target_col
                        )

                        if not is_valid_business:
                            tier = "Rejected"
                        else:
                            if confidence >= 0.80:
                                tier = "Confirmed"
                            elif confidence >= 0.65:
                                tier = "Likely"
                            elif confidence >= 0.50:
                                tier = "Possible"
                            else:
                                tier = "Rejected"

                        # Only Confirmed and Likely relationships should be returned and used by downstream modules
                        if tier not in ("Confirmed", "Likely"):
                            continue

                        # Prepare additional metadata for report improvements
                        confidence_breakdown = {
                            "semantic_similarity": round(
                                semantic_sim * semantic_weight, 4
                            ),
                            "referential_integrity": round(
                                referential * referential_weight, 4
                            ),
                            "cardinality": round(
                                cardinality_score * cardinality_weight, 4
                            ),
                            "datatype": round(datatype_score * datatype_weight, 4),
                            "value_overlap": round(overlap * overlap_weight, 4),
                        }

                        business_justification = (
                            f"Joins {source_id} ({source_col}) and {target_id} ({target_col}) based on matching {cardinality} key attributes. "
                            f"This allows enriching {source_id} records with related {target_id} features."
                        )

                        if cardinality == "many_to_one":
                            merge_risk = "Low. Safe many-to-one join that preserves the grain of the source dataset."
                        elif cardinality == "one_to_one":
                            merge_risk = "Low. Safe one-to-one join that does not alter dataset grain."
                        elif cardinality == "one_to_many":
                            merge_risk = "Medium. One-to-many join will duplicate source rows. Consider aggregation before join."
                        else:
                            merge_risk = "High. Many-to-many join will cause cartesian product expansion of rows. Perform aggregation or use a bridge table."

                        join_type = "inner" if overlap >= 0.95 else "left"
                        sql_join_example = (
                            f"SELECT *\n"
                            f"FROM {source_id} s\n"
                            f"{join_type.upper()} JOIN {target_id} t\n"
                            f"  ON s.{source_col} = t.{target_col}"
                        )

                        name_match = (
                            source_col.lower() == target_col.lower()
                            or source_col.lower().replace("_id", "")
                            in target_col.lower()
                            or target_col.lower().replace("_id", "")
                            in source_col.lower()
                        )
                        reason = self._build_reason(
                            overlap,
                            referential,
                            cardinality,
                            name_match,
                        )

                        relationship = Relationship(
                            source_dataset=source_id,
                            source_column=source_col,
                            target_dataset=target_id,
                            target_column=target_col,
                            relationship_type="foreign_key",
                            join_confidence=round(confidence, 4),
                            overlap_ratio=round(overlap, 4),
                            optional=bool(source_frame[column].isna().any()),
                            required=overlap >= 0.95
                            and not source_frame[column].isna().any(),
                            cardinality=cardinality,
                            join_type=join_type,
                            reason=reason,
                            referential_integrity=round(referential, 4),
                            tier=tier,
                            confidence_breakdown=confidence_breakdown,
                            business_justification=business_justification,
                            merge_risk=merge_risk,
                            sql_join_example=sql_join_example,
                        )
                        relationships.append(relationship)
                        if (source_id, source_col) not in foreign_key_map:
                            foreign_key_map[(source_id, source_col)] = (
                                target_id,
                                target_col,
                            )

        relationships.sort(key=lambda item: item.join_confidence, reverse=True)
        logger.info(
            "relationship_detection_end",
            relationships=len(relationships),
            duration_seconds=round(time.perf_counter() - start, 4),
        )
        return relationships, foreign_key_map

    def _is_candidate(
        self,
        source_id: str,
        source_col: str,
        target_id: str,
        target_col: str,
        profile_map: dict[tuple[str, str], ColumnProfile],
    ) -> bool:
        src_profile = profile_map.get((source_id, source_col))
        tgt_profile = profile_map.get((target_id, target_col))
        if not src_profile or not tgt_profile:
            return False

        src_dtype = src_profile.inferred_dtype
        tgt_dtype = tgt_profile.inferred_dtype

        # Check logical dtypes compatibility
        is_src_num = src_dtype in ("integer", "float")
        is_tgt_num = tgt_dtype in ("integer", "float")
        if is_src_num != is_tgt_num:
            return False
        if src_dtype == "boolean" and tgt_dtype != "boolean":
            return False
        if src_dtype == "datetime" and tgt_dtype != "datetime":
            return False
        if src_dtype == "string" and tgt_dtype != "string":
            return False

        # Target uniqueness and nullability (Primary Key characteristics)
        if tgt_profile.unique_percentage < 85:
            return False
        if tgt_profile.null_percentage > 10:
            return False

        # Column names and *_id / id prefixes check
        def is_key_name(name: str) -> bool:
            n = name.lower()
            return (
                n == "id"
                or n.endswith("_id")
                or n.startswith("id_")
                or n in {"uuid", "guid", "key", "code"}
                or "id" in n.split("_")
            )

        if not (is_key_name(source_col) or is_key_name(target_col)):
            return False

        return True

    def _semantic_similarity(
        self,
        source_id: str,
        source_col: str,
        target_id: str,
        target_col: str,
    ) -> float:
        s_col = source_col.lower()
        t_col = target_col.lower()
        if s_col == t_col:
            if s_col == "id":
                return 0.1  # id -> id across different datasets is semantically weak
            return 1.0

        def get_entity(col: str) -> str:
            if col.endswith("_id"):
                return col[:-3]
            if col.startswith("id_"):
                return col[3:]
            return col

        def singular(s: str) -> str:
            s = s.lower()
            if s.endswith("s"):
                return s[:-1]
            return s

        src_ent = get_entity(s_col)
        tgt_ent = get_entity(t_col)

        # If one is id and the other is entity_id
        if s_col == "id" or t_col == "id":
            other_ent = tgt_ent if s_col == "id" else src_ent
            id_ds = source_id if s_col == "id" else target_id
            if singular(other_ent) == singular(id_ds):
                return 1.0
            if singular(other_ent) in singular(id_ds) or singular(id_ds) in singular(other_ent):
                return 0.8
            return 0.0

        # If both are entity_id (e.g. restaurant_id -> rider_id)
        if src_ent and tgt_ent and singular(src_ent) == singular(tgt_ent):
            return 1.0

        return 0.0

    def _validate_business_sense(
        self,
        source_id: str,
        source_col: str,
        target_id: str,
        target_col: str,
    ) -> bool:
        # 1. Reject if either column is in the hardcoded list of descriptive business attributes
        rejected_attrs = {
            "avg_rating",
            "completed_orders",
            "prep_capacity",
            "shift_hours",
            "order_size",
        }
        if source_col.lower() in rejected_attrs or target_col.lower() in rejected_attrs:
            return False

        # 2. Enforce that both columns must be ID-like
        def is_id_like(col: str) -> bool:
            c = col.lower()
            return (
                c == "id"
                or c.endswith("_id")
                or c.startswith("id_")
                or c in {"uuid", "guid", "key", "code"}
                or "id" in c.split("_")
            )

        if not (is_id_like(source_col) and is_id_like(target_col)):
            return False

        # 3. Reject joining id -> id across different datasets
        if source_col.lower() == "id" and target_col.lower() == "id":
            return False

        # 4. Semantic alignment check:
        # Case A: One is "id" and the other is "[entity]_id"
        # The [entity] prefix of the "[entity]_id" column must match the dataset name of the "id" column.
        def get_entity_prefix(col: str) -> str | None:
            c = col.lower()
            if c.endswith("_id"):
                return c[:-3]
            if c.startswith("id_"):
                return c[3:]
            return None

        def singular(s: str) -> str:
            s = s.lower()
            if s.endswith("s"):
                return s[:-1]
            return s

        src_prefix = get_entity_prefix(source_col)
        tgt_prefix = get_entity_prefix(target_col)

        if source_col.lower() == "id" and tgt_prefix is not None:
            src_ds_sing = singular(source_id)
            tgt_pref_sing = singular(tgt_prefix)
            if (
                src_ds_sing != tgt_pref_sing
                and src_ds_sing not in tgt_pref_sing
                and tgt_pref_sing not in src_ds_sing
            ):
                return False

        elif target_col.lower() == "id" and src_prefix is not None:
            tgt_ds_sing = singular(target_id)
            src_pref_sing = singular(src_prefix)
            if (
                tgt_ds_sing != src_pref_sing
                and tgt_ds_sing not in src_pref_sing
                and src_pref_sing not in tgt_ds_sing
            ):
                return False

        elif src_prefix is not None and tgt_prefix is not None:
            src_pref_sing = singular(src_prefix)
            tgt_pref_sing = singular(tgt_prefix)
            if src_pref_sing != tgt_pref_sing:
                return False

        return True

    def _detect_primary_keys(self, profiles: list[DatasetProfile]) -> set[tuple[str, str]]:
        """Identify likely primary key columns."""
        keys: set[tuple[str, str]] = set()
        for profile in profiles:
            for column in profile.columns:
                if "identifier" in column.roles and (column.unique_percentage >= 90 or column.name.lower() == "id"):
                    keys.add((profile.dataset_id, column.name))
        return keys

    @staticmethod
    def _normalized_values(series: pd.Series) -> pd.Series:
        """Normalize a column to comparable string values."""
        if pd.api.types.is_numeric_dtype(series):
            vals = series.dropna()
            def norm(val):
                try:
                    f = float(val)
                    if f.is_integer():
                        return str(int(f))
                    return str(f)
                except (ValueError, TypeError):
                    return str(val).strip()
            return vals.map(norm)
        return series.dropna().astype(str).str.strip().map(
            lambda s: s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s
        )

    @staticmethod
    def _overlap_ratio(left: pd.Series, right: pd.Series) -> float:
        """Compute the fraction of left values present in right."""
        left_unique = set(left.unique())
        right_unique = set(right.unique())
        if not left_unique:
            return 0.0
        return len(left_unique & right_unique) / len(left_unique)

    @staticmethod
    def _referential_integrity(left: pd.Series, right: pd.Series) -> float:
        """Estimate referential integrity as 1 minus orphan rate."""
        left_unique = set(left.unique())
        right_unique = set(right.unique())
        if not left_unique:
            return 0.0
        orphans = left_unique - right_unique
        return 1.0 - (len(orphans) / len(left_unique))

    @staticmethod
    def _uniqueness_score(left: pd.Series, right: pd.Series) -> float:
        """Score how well uniqueness patterns align for a join."""
        left_ratio = left.nunique() / len(left)
        right_ratio = right.nunique() / len(right)
        return 1.0 - abs(left_ratio - right_ratio)

    @staticmethod
    def _name_compatible(source_col: str, target_col: str, target_dataset: str) -> bool:
        """Check whether column names suggest a plausible join."""
        source = source_col.lower()
        target = target_col.lower()
        if source == target:
            return True
        if source.endswith("_id") and target.endswith("_id"):
            return True
        dataset_token = target_dataset.split("_")[-1]
        if source.endswith("_id") and dataset_token in source:
            return target.endswith("_id") or target == "id"
        return False

    def _name_similarity_boost(self, source_col: str, target_col: str) -> float:
        """Boost confidence when names are strongly aligned."""
        if source_col.lower() == target_col.lower():
            return self._config.name_boost
        if source_col.lower().replace("_id", "") in target_col.lower():
            return self._config.name_boost / 2
        return 0.0

    @staticmethod
    def _infer_cardinality(
        source_values: pd.Series,
        target_values: pd.Series,
        target_is_pk: bool,
    ) -> str:
        """Infer relationship cardinality from uniqueness patterns."""
        source_unique_ratio = source_values.nunique() / len(source_values)
        target_unique_ratio = target_values.nunique() / len(target_values)

        if target_is_pk and source_unique_ratio < 0.95:
            return "many_to_one"
        if source_unique_ratio > 0.95 and target_unique_ratio > 0.95:
            return "one_to_one"
        if source_unique_ratio > 0.95 and target_unique_ratio < 0.95:
            return "one_to_many"
        if source_unique_ratio < 0.95 and target_unique_ratio < 0.95:
            return "many_to_many"
        return "many_to_one"

    @staticmethod
    def _build_reason(
        overlap: float,
        referential: float,
        cardinality: str,
        name_match: bool,
    ) -> str:
        """Build a human-readable relationship explanation."""
        parts = [
            f"{overlap:.0%} value overlap",
            f"{referential:.0%} referential integrity",
            f"{cardinality} cardinality",
        ]
        if name_match:
            parts.append("compatible column naming")
        return "; ".join(parts)

    @staticmethod
    def to_registry(relationships: list[Relationship]) -> dict[str, Any]:
        """Serialize relationships for report output."""
        return {
            "relationship_count": len(relationships),
            "relationships": [
                {
                    "source_dataset": rel.source_dataset,
                    "source_column": rel.source_column,
                    "target_dataset": rel.target_dataset,
                    "target_column": rel.target_column,
                    "relationship_type": rel.relationship_type,
                    "join_confidence": rel.join_confidence,
                    "overlap_ratio": rel.overlap_ratio,
                    "referential_integrity": rel.referential_integrity,
                    "optional": rel.optional,
                    "required": rel.required,
                    "cardinality": rel.cardinality,
                    "join_type": rel.join_type,
                    "reason": rel.reason,
                    "tier": getattr(rel, "tier", "possible"),
                    "confidence_breakdown": getattr(rel, "confidence_breakdown", {}),
                    "business_justification": getattr(
                        rel, "business_justification", ""
                    ),
                    "merge_risk": getattr(rel, "merge_risk", ""),
                    "sql_join_example": getattr(rel, "sql_join_example", ""),
                }
                for rel in relationships
            ],
        }
