"""Decision Intelligence Engine orchestrator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ml.decision.approval_manifest import ApprovalManifestBuilder
from ml.decision.confidence_engine import ConfidenceEngine
from ml.decision.config import DEFAULT_DECISION_CONFIG, DecisionConfig
from ml.decision.impact_estimator import ImpactEstimator
from ml.decision.logging_config import get_logger
from ml.decision.recommendation_engine import RecommendationEngine
from ml.decision.report_generator import DecisionReportGenerator
from ml.decision.rule_engine import RuleEngine
from ml.decision.utils import load_report_json

logger = get_logger(__name__)


class DecisionEngine:
    """Orchestrates the decision intelligence pipeline."""

    def __init__(self, config: DecisionConfig = DEFAULT_DECISION_CONFIG) -> None:
        """Initialize the decision engine.

        Args:
            config: Configuration settings.
        """
        self._config = config
        self._rule_engine = RuleEngine()
        self._recommendation_engine = RecommendationEngine(
            null_drop_threshold=config.null_drop_threshold
        )
        self._confidence_engine = ConfidenceEngine()
        self._impact_estimator = ImpactEstimator()
        self._manifest_builder = ApprovalManifestBuilder()

    def run(self) -> dict[str, Path]:
        """Execute the complete decision intelligence pipeline.

        Returns:
            dict[str, Path]: Map of generated report paths.
        """
        logger.info("decision_pipeline_start", reports_dir=str(self._config.reports_dir))
        start_time = time.perf_counter()

        # Load inputs
        reports = self._load_input_reports()

        # Run Rule Engine
        issues = self._rule_engine.analyze(reports)

        # Run Recommendation Engine
        raw_recs = self._recommendation_engine.generate(issues, reports)

        # Run Confidence Engine
        recs = self._confidence_engine.process(raw_recs, reports)

        # Run Impact Estimator
        impacts = self._impact_estimator.estimate(reports, recs)

        # Run Approval Manifest Builder
        manifest = self._manifest_builder.build(recs)

        # Generate Reports
        generator = DecisionReportGenerator(self._config.reports_dir)
        paths = generator.generate(recs, manifest, impacts, len(issues))

        duration = round(time.perf_counter() - start_time, 4)
        logger.info("decision_pipeline_end", duration_seconds=duration, files=len(paths))
        return paths

    def _load_input_reports(self) -> dict[str, Any]:
        """Load all dependent reports from the reports directory."""
        dir_path = self._config.reports_dir
        return {
            "validation_report": load_report_json(dir_path, "validation_report.json"),
            "dataset_profile": load_report_json(dir_path, "dataset_profile.json"),
            "schema_registry": load_report_json(dir_path, "schema_registry.json"),
            "relationship_registry": load_report_json(dir_path, "relationship_registry.json"),
            "feature_candidates": load_report_json(dir_path, "feature_candidates.json"),
            "target_candidates": load_report_json(dir_path, "target_candidates.json"),
            "leakage_report": load_report_json(dir_path, "leakage_report.json"),
            "intelligence_score": load_report_json(dir_path, "intelligence_score.json"),
        }
