"""Dataset Intelligence Engine orchestration and CLI."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from ml.intelligence.column_classifier import ColumnClassifier
from ml.intelligence.config import DEFAULT_CONFIG, IntelligenceConfig
from ml.intelligence.dataset_scanner import DatasetScanner
from ml.intelligence.feature_recommender import FeatureRecommender
from ml.intelligence.intelligence_score import IntelligenceScoreCalculator
from ml.intelligence.leakage_detector import LeakageDetector
from ml.intelligence.logging_config import configure_intelligence_logging, get_logger
from ml.intelligence.merge_strategy import MergeStrategyBuilder
from ml.intelligence.metadata_cache import ColumnMetadataCache
from ml.intelligence.models import IntelligenceReport
from ml.intelligence.relationship_detector import RelationshipDetector
from ml.intelligence.report_generator import IntelligenceReportGenerator
from ml.intelligence.schema_registry import SchemaRegistry
from ml.intelligence.statistics_engine import StatisticsEngine
from ml.intelligence.target_detector import TargetDetector
from ml.intelligence.version_tracker import VersionTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "ml" / "data" / "raw"
DEFAULT_REPORTS_DIR = REPO_ROOT / "ml" / "reports"

logger = get_logger(__name__)


def _resolve_path(path: Path) -> Path:
    """Resolve a path relative to the repository root when not absolute."""
    return path if path.is_absolute() else REPO_ROOT / path


class IntelligenceEngine:
    """Coordinates the full dataset intelligence pipeline."""

    def __init__(
        self,
        data_dir: Path,
        reports_dir: Path,
        config: IntelligenceConfig = DEFAULT_CONFIG,
    ) -> None:
        """Initialize the intelligence engine.

        Args:
            data_dir: Root directory to scan for CSV files.
            reports_dir: Directory for generated reports.
            config: Intelligence configuration.
        """
        self._data_dir = data_dir
        self._reports_dir = reports_dir
        self._config = config
        self._cache = ColumnMetadataCache(config=config)

    def run(self) -> IntelligenceReport:
        """Execute the full intelligence pipeline."""
        report = IntelligenceReport(
            started_at=datetime.now(UTC),
            data_dir=str(self._data_dir),
            reports_dir=str(self._reports_dir),
        )

        scanned, frames = DatasetScanner(
            self._data_dir,
            self._config.sample_rows,
        ).scan()
        if not frames:
            logger.warning("no_datasets_found", data_dir=str(self._data_dir))
            report.finished_at = datetime.now(UTC)
            return report

        report.schema_registry = SchemaRegistry(self._cache).build(frames, scanned)
        report.version_report = VersionTracker().compare(
            report.schema_registry,
            self._reports_dir,
        )

        profiles = StatisticsEngine(self._cache, self._config).profile_all(
            frames, scanned
        )
        profiles = ColumnClassifier(self._config).classify_profiles(profiles)

        relationships, fk_map = RelationshipDetector(self._config).detect(
            frames, profiles
        )
        report.relationships = relationships
        ColumnClassifier.mark_foreign_keys(profiles, fk_map)

        report.target_candidates = TargetDetector(self._config).detect(profiles)
        report.leakage_findings = LeakageDetector(self._config).detect(
            frames,
            profiles,
            report.target_candidates,
        )
        LeakageDetector.annotate_profiles(profiles, report.leakage_findings)

        report.feature_candidates = FeatureRecommender(self._config).recommend(profiles)
        report.datasets = profiles
        report.merge_strategies = MergeStrategyBuilder().build(relationships)
        report.intelligence_score = IntelligenceScoreCalculator(self._config).calculate(
            report
        )
        report.finished_at = datetime.now(UTC)
        return report


def print_summary(report: IntelligenceReport, paths: dict[str, Path]) -> None:
    """Print a human-readable intelligence summary."""
    print("\n" + "=" * 60)
    print("ETAIQ DATASET INTELLIGENCE SUMMARY")
    print("=" * 60)
    print(f"Datasets discovered: {len(report.datasets)}")
    print(f"Relationships detected: {len(report.relationships)}")
    strong_targets = [t for t in report.target_candidates if t.tier == "strong"]
    print(f"Strong target candidates: {len(strong_targets)}")
    print(f"Total target candidates: {len(report.target_candidates)}")
    print(f"Leakage findings: {len(report.leakage_findings)}")
    print(f"Feature recommendations: {len(report.feature_candidates)}")
    if report.intelligence_score:
        print(
            f"Overall Intelligence Score: "
            f"{report.intelligence_score.get('overall_intelligence_score', 'N/A')}"
        )
    if strong_targets:
        top = strong_targets[0]
        print(
            f"Top strong target: {top.dataset_id}.{top.column} (confidence={top.confidence})"
        )
    elif report.target_candidates:
        top = report.target_candidates[0]
        print(
            f"Top target candidate: {top.dataset_id}.{top.column} (confidence={top.confidence})"
        )
    print(f"Version status: {report.version_report.get('status', 'unknown')}")
    print("\nReports:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")
    print("=" * 60 + "\n")


def run_intelligence(data_dir: Path, reports_dir: Path) -> int:
    """Run intelligence pipeline end to end."""
    configure_intelligence_logging()
    logger.info(
        "intelligence_pipeline_start",
        data_dir=str(data_dir),
        reports_dir=str(reports_dir),
    )
    start = time.perf_counter()

    engine = IntelligenceEngine(data_dir, reports_dir)
    report = engine.run()
    paths = IntelligenceReportGenerator(reports_dir).generate(report)
    print_summary(report, paths)

    logger.info(
        "intelligence_pipeline_end",
        datasets=len(report.datasets),
        duration_seconds=round(time.perf_counter() - start, 4),
    )
    return 0 if report.datasets else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ETAIQ Dataset Intelligence Engine",
        prog="python -m ml.intelligence.main",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory to scan recursively for CSV files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Directory for intelligence reports (default: {DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_CONFIG.sample_rows,
        help=f"Maximum rows to profile for large CSV files (default: {DEFAULT_CONFIG.sample_rows})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    parser = build_parser()
    args = parser.parse_args(argv)
    data_dir = _resolve_path(args.data_dir)
    reports_dir = _resolve_path(args.reports_dir)

    if not data_dir.is_dir():
        print(f"ERROR: Data directory does not exist: {data_dir}", file=sys.stderr)
        return 2

    return run_intelligence(data_dir, reports_dir)


if __name__ == "__main__":
    raise SystemExit(main())
