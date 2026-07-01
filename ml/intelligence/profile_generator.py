"""Dataset profile document generation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ml.intelligence.logging_config import get_logger
from ml.intelligence.models import DatasetProfile

logger = get_logger(__name__)


class ProfileGenerator:
    """Builds dataset profile payloads for reporting."""

    def build_json(self, profiles: list[DatasetProfile]) -> dict[str, Any]:
        """Serialize dataset profiles to a JSON-compatible dictionary.

        Args:
            profiles: Dataset profiles.

        Returns:
            dict[str, Any]: Profile document.
        """
        logger.info("profile_generation_start", datasets=len(profiles))
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_count": len(profiles),
            "datasets": [self._dataset_to_dict(profile) for profile in profiles],
        }
        logger.info("profile_generation_end", datasets=len(profiles))
        return payload

    def build_markdown(self, profiles: list[DatasetProfile]) -> str:
        """Render dataset profiles as Markdown.

        Args:
            profiles: Dataset profiles.

        Returns:
            str: Markdown document.
        """
        lines = [
            "# ETAIQ Dataset Profile",
            "",
            f"**Generated:** {datetime.now(UTC).isoformat()}",
            f"**Datasets profiled:** {len(profiles)}",
            "",
        ]

        for profile in profiles:
            lines.extend(
                [
                    f"## {profile.dataset_id}",
                    "",
                    f"- **File:** `{profile.filename}`",
                    f"- **Path:** `{profile.relative_path}`",
                    f"- **Rows:** {profile.row_count:,}",
                    f"- **Columns:** {profile.column_count}",
                    f"- **Memory:** {profile.memory_bytes:,} bytes",
                    f"- **Duplicate rows (sample):** {profile.duplicate_row_count}",
                    "",
                    "### Columns",
                    "",
                    "| Column | Type | Null % | Unique % | Roles |",
                    "|--------|------|-------:|---------:|-------|",
                ]
            )
            for column in profile.columns:
                roles = ", ".join(column.roles)
                lines.append(
                    f"| {column.name} | {column.inferred_dtype} | "
                    f"{column.null_percentage:.1f} | {column.unique_percentage:.1f} | {roles} |"
                )
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _dataset_to_dict(profile: DatasetProfile) -> dict[str, Any]:
        """Convert a dataset profile to a dictionary.

        Args:
            profile: Dataset profile.

        Returns:
            dict[str, Any]: Serialized dataset profile.
        """
        return {
            "dataset_id": profile.dataset_id,
            "filename": profile.filename,
            "relative_path": profile.relative_path,
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "memory_bytes": profile.memory_bytes,
            "duplicate_row_count": profile.duplicate_row_count,
            "column_groups": profile.column_groups,
            "columns": [
                {
                    "name": column.name,
                    "inferred_dtype": column.inferred_dtype,
                    "pandas_dtype": column.pandas_dtype,
                    "null_count": column.null_count,
                    "null_percentage": column.null_percentage,
                    "unique_count": column.unique_count,
                    "unique_percentage": column.unique_percentage,
                    "duplicate_rows": column.duplicate_rows,
                    "sample_values": column.sample_values,
                    "roles": column.roles,
                    "statistics": column.statistics,
                    "flags": {
                        "numeric": column.is_numeric,
                        "categorical": column.is_categorical,
                        "boolean": column.is_boolean,
                        "datetime": column.is_datetime,
                        "text": column.is_text,
                        "high_cardinality": column.is_high_cardinality,
                    },
                }
                for column in profile.columns
            ],
        }
