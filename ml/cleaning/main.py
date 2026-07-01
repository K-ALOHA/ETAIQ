"""CLI main entry point for the ETAIQ Cleaning Execution Engine."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ml.cleaning.cleaning_engine import CleaningEngine
from ml.cleaning.config import DEFAULT_CLEANING_CONFIG, CleaningConfig
from ml.cleaning.logging_config import configure_cleaning_logging, get_logger
from ml.cleaning.rollback import RollbackManager

REPO_ROOT = Path(__file__).resolve().parents[2]
logger = get_logger(__name__)


def _resolve_path(path: Path) -> Path:
    """Resolve a path relative to the repository root when not absolute."""
    return path if path.is_absolute() else REPO_ROOT / path


def print_summary(paths: dict[str, Path]) -> None:
    """Print execution results cleanly to stdout."""
    print("\n" + "=" * 60)
    print("ETAIQ CLEANING EXECUTION PIPELINE COMPLETED")
    print("=" * 60)
    print("Generated Reports and Manifests:")
    for name, path in sorted(paths.items()):
        print(f"  - {name:25s}: {path}")
    print("=" * 60 + "\n")


def execute_rollback(reports_dir: Path) -> int:
    """Trigger restoration backup workflow."""
    configure_cleaning_logging()
    logger.info("rollback_cli_start", reports_dir=str(reports_dir))
    
    manifest_name = DEFAULT_CLEANING_CONFIG.rollback_manifest_filename
    rollback_mgr = RollbackManager(reports_dir / manifest_name)
    success = rollback_mgr.execute_rollback()

    if success:
        print("\n" + "=" * 60)
        print("ETAIQ ROLLBACK SUCCESSFUL - DATA RESTORED TO RAW BACKUP")
        print("=" * 60 + "\n")
        logger.info("rollback_cli_success")
        return 0
    else:
        print("\n" + "=" * 60)
        print("ERROR: ETAIQ ROLLBACK FAILED")
        print("=" * 60 + "\n")
        logger.error("rollback_cli_failed")
        return 1


def run_cleaning(
    raw_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    force_approve_all: bool,
) -> int:
    """Orchestrate and run standard clean sequence."""
    configure_cleaning_logging()
    logger.info(
        "cleaning_cli_start",
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        reports_dir=str(reports_dir),
        force_approve_all=force_approve_all,
    )
    start = time.perf_counter()

    config = CleaningConfig(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
    )
    engine = CleaningEngine(config=config)
    try:
        paths = engine.run(force_approve_all=force_approve_all)
        print_summary(paths)
        logger.info(
            "cleaning_cli_end",
            duration_seconds=round(time.perf_counter() - start, 4),
        )
        return 0
    except Exception as exc:
        print(f"ERROR executing cleaning engine: {exc}", file=sys.stderr)
        logger.exception("cleaning_cli_failed", error=str(exc))
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ETAIQ Cleaning Execution Engine CLI",
        prog="python -m ml.cleaning.main",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_CLEANING_CONFIG.raw_dir,
        help=f"Directory containing raw inputs (default: {DEFAULT_CLEANING_CONFIG.raw_dir})",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_CLEANING_CONFIG.processed_dir,
        help=f"Output directory for processed files (default: {DEFAULT_CLEANING_CONFIG.processed_dir})",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_CLEANING_CONFIG.reports_dir,
        help=f"Directory containing manifests/reports (default: {DEFAULT_CLEANING_CONFIG.reports_dir})",
    )
    parser.add_argument(
        "--force-approve-all",
        action="store_true",
        help="Execute all manifest decisions, ignoring status checking.",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Restore datasets in processed/ to their raw backup state.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main entry point."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    parser = build_parser()
    args = parser.parse_args(argv)

    raw_dir = _resolve_path(args.raw_dir)
    processed_dir = _resolve_path(args.processed_dir)
    reports_dir = _resolve_path(args.reports_dir)

    if args.rollback:
        if not reports_dir.is_dir():
            print(f"ERROR: Reports directory does not exist: {reports_dir}", file=sys.stderr)
            return 2
        return execute_rollback(reports_dir)

    if not raw_dir.is_dir():
        print(f"ERROR: Raw data directory does not exist: {raw_dir}", file=sys.stderr)
        return 2

    return run_cleaning(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        reports_dir=reports_dir,
        force_approve_all=args.force_approve_all,
    )


if __name__ == "__main__":
    raise SystemExit(main())
