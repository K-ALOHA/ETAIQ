"""CLI entry point for the ETAIQ data validation engine."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml.validation.logging_config import configure_validation_logging, get_logger
from ml.validation.report_generator import ReportGenerator
from ml.validation.schemas import ALL_SCHEMAS, SCHEMA_BY_NAME, DatasetSchema
from ml.validation.validator import ValidationEngine, load_dataframe

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "ml" / "data" / "raw"
DEFAULT_REPORTS_DIR = REPO_ROOT / "ml" / "reports"

logger = get_logger(__name__)


def _resolve_path(path: Path) -> Path:
    """Resolve a path relative to the repository root when not absolute.

    Args:
        path: User-supplied path.

    Returns:
        Path: Absolute resolved path.
    """
    return path if path.is_absolute() else REPO_ROOT / path


def load_datasets(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all expected CSV datasets from the raw data directory.

    Args:
        data_dir: Directory containing ``restaurants.csv``, ``riders.csv``, ``orders.csv``.

    Returns:
        dict[str, pd.DataFrame]: Loaded datasets keyed by schema name.

    Raises:
        FileNotFoundError: When a required CSV file is missing.
    """
    datasets: dict[str, pd.DataFrame] = {}
    for schema in ALL_SCHEMAS:
        csv_path = data_dir / schema.filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Required dataset not found: {csv_path}")
        logger.info("dataset_load_start", dataset=schema.name, path=str(csv_path))
        datasets[schema.name] = load_dataframe(csv_path)
        logger.info(
            "dataset_load_end",
            dataset=schema.name,
            rows=len(datasets[schema.name]),
            columns=len(datasets[schema.name].columns),
        )
    return datasets


def print_summary(summary: object, report_paths: dict[str, Path]) -> None:
    """Print a human-readable validation summary to stdout.

    Args:
        summary: ValidationSummary with results and quality score.
        report_paths: Paths to generated report files.
    """
    from ml.validation.models import ValidationSummary

    assert isinstance(summary, ValidationSummary)

    passed = sum(1 for r in summary.results if r.passed)
    failed = len(summary.results) - passed

    print("\n" + "=" * 60)
    print("ETAIQ DATA VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Quality Score: {summary.quality_score:.2f} / 100")
    print(f"Checks Passed: {passed} / {len(summary.results)}")
    print(f"Checks Failed: {failed}")
    print("\nComponent Scores:")
    for category, score in summary.component_scores.items():
        print(f"  - {category:12s}: {score:.2f}")
    print("\nReports:")
    for name, path in report_paths.items():
        print(f"  - {name}: {path}")
    print("=" * 60 + "\n")


def run_validation(data_dir: Path, reports_dir: Path) -> int:
    """Execute the full validation pipeline.

    Args:
        data_dir: Directory containing raw CSV files.
        reports_dir: Directory for generated reports.

    Returns:
        int: Process exit code (0 = success, 1 = validation failures).
    """
    configure_validation_logging()
    logger.info(
        "validation_pipeline_start",
        data_dir=str(data_dir),
        reports_dir=str(reports_dir),
    )

    started = datetime.now(UTC)
    datasets = load_datasets(data_dir)
    schemas: dict[str, DatasetSchema] = {
        name: SCHEMA_BY_NAME[name] for name in datasets
    }

    engine = ValidationEngine()
    summary = engine.run(datasets, schemas)
    summary.started_at = started
    summary.finished_at = datetime.now(UTC)
    summary.data_dir = str(data_dir)
    summary.reports_dir = str(reports_dir)

    generator = ReportGenerator(reports_dir)
    report_paths = generator.generate(summary)

    print_summary(summary, report_paths)

    logger.info(
        "validation_pipeline_end",
        quality_score=summary.quality_score,
        passed=sum(1 for r in summary.results if r.passed),
        failed=sum(1 for r in summary.results if not r.passed),
        duration_seconds=round(
            (summary.finished_at - summary.started_at).total_seconds(), 2
        ),
    )

    has_failures = any(not result.passed for result in summary.results)
    return 1 if has_failures else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        description="ETAIQ raw data validation engine",
        prog="python -m ml.validation.main",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing raw CSV files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Output directory for validation reports (default: {DEFAULT_REPORTS_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main entry point.

    Args:
        argv: Optional argument list override for testing.

    Returns:
        int: Process exit code.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    parser = build_parser()
    args = parser.parse_args(argv)

    data_dir = _resolve_path(args.data_dir)
    reports_dir = _resolve_path(args.reports_dir)

    if not data_dir.is_dir():
        print(f"ERROR: Data directory does not exist: {data_dir}", file=sys.stderr)
        return 2

    try:
        return run_validation(data_dir, reports_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
