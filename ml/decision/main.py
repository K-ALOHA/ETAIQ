"""CLI entry point for the ETAIQ Decision Intelligence Engine."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ml.decision.config import DecisionConfig
from ml.decision.decision_engine import DecisionEngine
from ml.decision.logging_config import configure_decision_logging, get_logger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = REPO_ROOT / "ml" / "reports"

logger = get_logger(__name__)


def _resolve_path(path: Path) -> Path:
    """Resolve a path relative to the repository root when not absolute."""
    return path if path.is_absolute() else REPO_ROOT / path


def print_summary(paths: dict[str, Path]) -> None:
    """Print a beautiful execution summary to stdout."""
    print("\n" + "=" * 60)
    print("ETAIQ DECISION INTELLIGENCE PIPELINE COMPLETED")
    print("=" * 60)
    print("Generated Reports and Manifests:")
    for name, path in sorted(paths.items()):
        print(f"  - {name:25s}: {path}")
    print("=" * 60 + "\n")


def run_decision(reports_dir: Path) -> int:
    """Run decision intelligence pipeline end-to-end.

    Args:
        reports_dir: Directory containing input reports and where output will be saved.

    Returns:
        int: CLI exit code.
    """
    configure_decision_logging()
    logger.info("decision_cli_run_start", reports_dir=str(reports_dir))
    start = time.perf_counter()

    config = DecisionConfig(reports_dir=reports_dir)
    engine = DecisionEngine(config=config)
    try:
        paths = engine.run()
        print_summary(paths)
        logger.info(
            "decision_cli_run_end",
            duration_seconds=round(time.perf_counter() - start, 4),
        )
        return 0
    except Exception as exc:
        print(f"ERROR executing decision engine: {exc}", file=sys.stderr)
        logger.exception("decision_cli_failed", error=str(exc))
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ETAIQ Decision Intelligence Engine CLI",
        prog="python -m ml.decision.main",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Directory containing inputs and outputs (default: {DEFAULT_REPORTS_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main entry point."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    parser = build_parser()
    args = parser.parse_args(argv)

    reports_dir = _resolve_path(args.reports_dir)
    if not reports_dir.is_dir():
        print(f"ERROR: Reports directory does not exist: {reports_dir}", file=sys.stderr)
        return 2

    return run_decision(reports_dir)


if __name__ == "__main__":
    raise SystemExit(main())
