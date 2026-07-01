"""Merge and join strategy recommendations."""

from __future__ import annotations

import time
from typing import Any

from ml.intelligence.logging_config import get_logger
from ml.intelligence.models import Relationship

logger = get_logger(__name__)


class MergeStrategyBuilder:
    """Builds recommended dataset merge strategies from detected relationships."""

    def build(self, relationships: list[Relationship]) -> list[dict[str, Any]]:
        """Create merge strategy recommendations.

        Args:
            relationships: Detected relationships between datasets.

        Returns:
            list[dict[str, Any]]: Merge strategy entries.
        """
        logger.info("merge_strategy_build_start", relationships=len(relationships))
        start = time.perf_counter()

        strategies: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()

        for rel in relationships:
            key = (
                rel.source_dataset,
                rel.source_column,
                rel.target_dataset,
                rel.target_column,
            )
            if key in seen:
                continue
            seen.add(key)

            join_type = "inner" if rel.required else "left"
            strategies.append(
                {
                    "left_dataset": rel.source_dataset,
                    "right_dataset": rel.target_dataset,
                    "left_on": rel.source_column,
                    "right_on": rel.target_column,
                    "join_type": join_type,
                    "optional": rel.optional,
                    "required": rel.required,
                    "confidence": rel.join_confidence,
                    "cardinality": rel.cardinality,
                    "strategy": self._strategy_label(rel),
                    "rationale": self._strategy_rationale(rel),
                }
            )

        strategies.sort(key=lambda item: item["confidence"], reverse=True)
        logger.info(
            "merge_strategy_build_end",
            strategies=len(strategies),
            duration_seconds=round(time.perf_counter() - start, 4),
        )
        return strategies

    @staticmethod
    def _strategy_label(rel: Relationship) -> str:
        """Return a short strategy label for a relationship.

        Args:
            rel: Detected relationship.

        Returns:
            str: Strategy label.
        """
        if rel.cardinality == "many_to_one":
            return "enrich_source_with_target"
        if rel.cardinality == "one_to_many":
            return "aggregate_target_to_source"
        if rel.cardinality == "many_to_many":
            return "bridge_or_aggregate_before_join"
        return "direct_join"

    @staticmethod
    def _strategy_rationale(rel: Relationship) -> str:
        """Explain why a merge strategy is recommended.

        Args:
            rel: Detected relationship.

        Returns:
            str: Human-readable rationale.
        """
        join = "required inner join" if rel.required else "optional left join"
        return (
            f"Detected {rel.cardinality} relationship with "
            f"{rel.overlap_ratio:.0%} overlap; recommend {join}."
        )
