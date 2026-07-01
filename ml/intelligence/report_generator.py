"""Report artifact writer for the Dataset Intelligence Engine."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.intelligence.logging_config import get_logger
from ml.intelligence.models import IntelligenceReport
from ml.intelligence.profile_generator import ProfileGenerator
from ml.intelligence.relationship_detector import RelationshipDetector

logger = get_logger(__name__)


class IntelligenceReportGenerator:
    """Writes all intelligence artifacts to the reports directory."""

    def __init__(self, reports_dir: Path) -> None:
        """Initialize the report generator."""
        self._reports_dir = reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._profile_generator = ProfileGenerator()

    def generate(self, report: IntelligenceReport) -> dict[str, Path]:
        """Write all intelligence report files."""
        logger.info(
            "intelligence_report_generation_start", reports_dir=str(self._reports_dir)
        )

        profile_json = self._profile_generator.build_json(report.datasets)
        profile_md = self._build_profile_markdown(report)

        paths = {
            "dataset_profile_json": self._write_json(
                "dataset_profile.json", profile_json
            ),
            "dataset_profile_md": self._write_text("dataset_profile.md", profile_md),
            "schema_registry_json": self._write_json(
                "schema_registry.json", report.schema_registry
            ),
            "relationship_registry_json": self._write_json(
                "relationship_registry.json",
                self._relationship_payload(report),
            ),
            "feature_candidates_json": self._write_json(
                "feature_candidates.json",
                self._feature_candidates_payload(report),
            ),
            "merge_strategy_json": self._write_json(
                "merge_strategy.json",
                self._merge_strategy_payload(report),
            ),
            "target_candidates_json": self._write_json(
                "target_candidates.json",
                self._target_candidates_payload(report),
            ),
            "leakage_report_json": self._write_json(
                "leakage_report.json",
                self._leakage_payload(report),
            ),
            "data_dictionary_md": self._write_text(
                "data_dictionary.md",
                self._build_data_dictionary(report),
            ),
            "version_report_json": self._write_json(
                "version_report.json", report.version_report
            ),
            "intelligence_score_json": self._write_json(
                "intelligence_score.json",
                report.intelligence_score,
            ),
            "relationship_graph_json": self._write_json(
                "relationship_graph.json",
                self._relationship_graph_payload(report),
            ),
        }

        logger.info("intelligence_report_generation_end", artifacts=len(paths))
        return paths

    def _write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        """Write a JSON artifact."""
        path = self._reports_dir / filename
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def _write_text(self, filename: str, content: str) -> Path:
        """Write a text artifact."""
        path = self._reports_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def _target_candidates_payload(self, report: IntelligenceReport) -> dict[str, Any]:
        """Serialize target candidates grouped by confidence tier."""
        grouped: dict[str, list[dict[str, Any]]] = {
            "strong_targets": [],
            "possible_targets": [],
            "weak_targets": [],
        }
        for item in report.target_candidates:
            entry = {
                "dataset_id": item.dataset_id,
                "column": item.column,
                "rank": item.rank,
                "confidence": item.confidence,
                "tier": item.tier,
                "target_type": item.target_type,
                "explanation": item.explanation,
                "evidence": item.evidence,
            }
            key = f"{item.tier}_targets"
            if key in grouped:
                grouped[key].append(entry)

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": {
                "total_candidates": len(report.target_candidates),
                "strong_count": len(grouped["strong_targets"]),
                "possible_count": len(grouped["possible_targets"]),
                "weak_count": len(grouped["weak_targets"]),
            },
            **grouped,
            "recommendations": self._target_recommendations(report),
            "next_steps": [
                "Confirm strong targets with domain experts.",
                "Exclude leakage columns before feature engineering.",
                "Use possible targets only after business validation.",
            ],
        }

    @staticmethod
    def _target_recommendations(report: IntelligenceReport) -> list[str]:
        """Build target-specific recommendations."""
        recs: list[str] = []
        strong = [t for t in report.target_candidates if t.tier == "strong"]
        if strong:
            top = strong[0]
            recs.append(
                f"Primary target recommendation: {top.dataset_id}.{top.column} "
                f"(confidence={top.confidence})."
            )
        if report.leakage_findings:
            recs.append(
                f"Review {len(report.leakage_findings)} leakage findings before modeling."
            )
        return recs

    def _feature_candidates_payload(self, report: IntelligenceReport) -> dict[str, Any]:
        """Serialize feature recommendations with classifications."""
        by_class: dict[str, list[dict[str, Any]]] = {}
        for item in report.feature_candidates:
            by_class.setdefault(item.classification, []).append(
                {
                    "dataset_id": item.dataset_id,
                    "column": item.column,
                    "classification": item.classification,
                    "recommendation": item.recommendation,
                    "confidence": item.confidence,
                    "reason": item.reason,
                    "encoding": item.encoding,
                    "scaling": item.scaling,
                    "engineering": item.engineering,
                }
            )

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": {
                "total_columns": len(report.feature_candidates),
                "by_classification": {k: len(v) for k, v in by_class.items()},
            },
            "classifications": by_class,
            "engineering_notes": [
                "Apply encoding and scaling suggestions per column classification.",
                "Derive calendar features from datetime columns.",
                "Drop identifier, metadata, PII, and leakage columns before training.",
            ],
            "warnings": self._feature_warnings(report),
        }

    @staticmethod
    def _feature_warnings(report: IntelligenceReport) -> list[str]:
        """Collect feature-related warnings."""
        warnings: list[str] = []
        leakage = [
            f for f in report.feature_candidates if f.classification == "leakage"
        ]
        if leakage:
            warnings.append(f"{len(leakage)} columns flagged as potential leakage.")
        weak = [
            f for f in report.feature_candidates if f.recommendation == "weak_feature"
        ]
        if weak:
            warnings.append(f"{len(weak)} columns classified as weak features.")
        return warnings

    def _relationship_payload(self, report: IntelligenceReport) -> dict[str, Any]:
        """Serialize relationships with confidence and reasons."""
        registry = RelationshipDetector.to_registry(report.relationships)
        registry["summary"] = {
            "relationship_count": len(report.relationships),
            "avg_confidence": round(
                sum(r.join_confidence for r in report.relationships)
                / max(len(report.relationships), 1),
                4,
            ),
        }
        registry["recommendations"] = [
            f"Use {r.join_type} join for {r.source_dataset}.{r.source_column} "
            f"→ {r.target_dataset}.{r.target_column} ({r.cardinality})."
            for r in report.relationships[:5]
        ]
        return registry

    def _relationship_graph_payload(self, report: IntelligenceReport) -> dict[str, Any]:
        """Generate payload for relationship_graph.json."""
        datasets = set()
        for rel in report.relationships:
            datasets.add(rel.source_dataset)
            datasets.add(rel.target_dataset)

        if not datasets and report.datasets:
            for p in report.datasets:
                datasets.add(p.dataset_id)

        nodes = [{"id": ds, "label": ds} for ds in sorted(datasets)]

        edges = []
        for rel in report.relationships:
            edges.append(
                {
                    "source": rel.source_dataset,
                    "target": rel.target_dataset,
                    "direction": f"{rel.source_dataset}.{rel.source_column} -> {rel.target_dataset}.{rel.target_column}",
                    "relationship_type": rel.relationship_type,
                    "confidence": rel.join_confidence,
                    "source_column": rel.source_column,
                    "target_column": rel.target_column,
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
        }

    @staticmethod
    def _merge_strategy_payload(report: IntelligenceReport) -> dict[str, Any]:
        """Serialize merge strategies with engineering notes."""
        return {
            "strategies": report.merge_strategies,
            "summary": {"strategy_count": len(report.merge_strategies)},
            "engineering_notes": [
                "Join datasets in confidence order starting with highest overlap.",
                "Prefer left joins for optional relationships.",
            ],
        }

    def _leakage_payload(self, report: IntelligenceReport) -> dict[str, Any]:
        """Serialize leakage findings with severity and recommendations."""
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": {
                "finding_count": len(report.leakage_findings),
                "high_severity": sum(
                    1 for f in report.leakage_findings if f.severity == "HIGH"
                ),
            },
            "findings": [
                {
                    "dataset_id": item.dataset_id,
                    "column": item.column,
                    "severity": item.severity,
                    "confidence": item.confidence,
                    "rationale": item.rationale,
                    "recommendation": item.recommendation,
                    "related_target": item.related_target,
                }
                for item in report.leakage_findings
            ],
            "warnings": [
                f"{item.column}: {item.severity} — {item.recommendation}"
                for item in report.leakage_findings
                if item.severity == "HIGH"
            ],
            "next_steps": [
                "Remove HIGH severity leakage columns before training.",
                "Review MEDIUM severity columns with domain experts.",
            ],
        }

    def _build_profile_markdown(self, report: IntelligenceReport) -> str:
        """Build enhanced dataset profile markdown."""
        base = self._profile_generator.build_markdown(report.datasets)
        score_section = ""
        if report.intelligence_score:
            score_section = (
                "\n## Intelligence Score\n\n"
                f"- **Overall:** {report.intelligence_score.get('overall_intelligence_score')}\n"
                f"- **Discovery:** {report.intelligence_score.get('discovery_quality')}\n"
                f"- **Relationships:** {report.intelligence_score.get('relationship_confidence')}\n"
                f"- **Schema:** {report.intelligence_score.get('schema_confidence')}\n"
                f"- **Features:** {report.intelligence_score.get('feature_confidence')}\n"
                f"- **Targets:** {report.intelligence_score.get('target_confidence')}\n"
            )
        return (
            base
            + score_section
            + "\n## Next Steps\n\n"
            + "- Review strong target candidates and leakage report.\n"
            + "- Apply feature classifications before cleaning pipeline.\n"
            + "- Validate join strategies with domain knowledge.\n"
        )

    def _build_data_dictionary(self, report: IntelligenceReport) -> str:
        """Build an enhanced cross-dataset data dictionary."""
        lines = [
            "# ETAIQ Data Dictionary",
            "",
            f"**Generated:** {datetime.now(UTC).isoformat()}",
            "",
            "## Summary",
            "",
            f"- Datasets: {len(report.datasets)}",
            f"- Relationships: {len(report.relationships)}",
            f"- Strong targets: {sum(1 for t in report.target_candidates if t.tier == 'strong')}",
            "",
        ]
        feature_lookup = {
            (f.dataset_id, f.column): f for f in report.feature_candidates
        }
        for profile in report.datasets:
            lines.extend([f"## {profile.dataset_id}", ""])
            for column in profile.columns:
                feat = feature_lookup.get((profile.dataset_id, column.name))
                lines.extend(
                    [
                        f"### `{column.name}`",
                        "",
                        f"- **Type:** {column.inferred_dtype}",
                        f"- **Roles:** {', '.join(column.roles)}",
                        f"- **Null %:** {column.null_percentage}",
                        f"- **Unique %:** {column.unique_percentage}",
                    ]
                )
                if feat:
                    lines.extend(
                        [
                            f"- **Classification:** {feat.classification}",
                            f"- **Confidence:** {feat.confidence}",
                            f"- **Reason:** {feat.reason}",
                            f"- **Recommendation:** {feat.rationale}",
                        ]
                    )
                lines.append("")
        lines.extend(
            [
                "## Engineering Notes",
                "",
                "- Drop identifier, metadata, PII, and leakage columns before training.",
                "- Apply encoding and scaling per feature_candidates.json.",
                "",
            ]
        )
        return "\n".join(lines)
