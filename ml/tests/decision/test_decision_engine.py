"""Unit and integration tests for the ETAIQ Decision Intelligence Engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.decision.approval_manifest import ApprovalManifestBuilder
from ml.decision.confidence_engine import ConfidenceEngine
from ml.decision.config import DecisionConfig
from ml.decision.decision_engine import DecisionEngine
from ml.decision.impact_estimator import ImpactEstimator
from ml.decision.main import main
from ml.decision.models import (
    ActionableIssue,
    ActionVerb,
    CleaningRecommendation,
    PriorityLevel,
)
from ml.decision.recommendation_engine import RecommendationEngine
from ml.decision.report_generator import DecisionReportGenerator
from ml.decision.rule_engine import RuleEngine


@pytest.fixture
def mock_reports_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with mock reports mimicking ETAIQ output."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    # 1. Validation Report
    val_report = {
        "quality_score": 66.62,
        "component_scores": {
            "schema": 25.0,
            "nulls": 99.31,
            "duplicates": 95.03,
            "gps": 100.0,
            "foreign_key": 0.0,
            "target": 100.0,
        },
        "results": [
            {
                "validator_name": "schema",
                "dataset_name": "restaurants",
                "passed": False,
                "score": 0.0,
                "duration_seconds": 0.0001,
                "details": {
                    "required_columns": ["restaurant_id", "city"],
                    "missing_columns": ["city"],
                    "extra_columns": ["cuisine"],
                },
            },
            {
                "validator_name": "nulls",
                "dataset_name": "restaurants",
                "passed": False,
                "score": 100.0,
                "details": {
                    "non_nullable_violations": ["avg_rating"],
                    "per_column": {"avg_rating": {"count": 343, "percentage": 8.04}},
                },
            },
            {
                "validator_name": "duplicates",
                "dataset_name": "restaurants",
                "passed": True,
                "score": 93.76,
                "details": {
                    "exact_duplicate_rows": 266,
                },
            },
            {
                "validator_name": "foreign_key",
                "dataset_name": "orders",
                "passed": False,
                "score": 0.0,
                "details": {
                    "per_column": {
                        "restaurant_id": {
                            "orphan_count": 307500,
                            "reference_dataset_missing": True,
                        }
                    }
                },
            },
        ],
    }

    # 2. Dataset Profile
    dataset_profile = {
        "datasets": [
            {
                "dataset_id": "restaurants",
                "columns": [
                    {
                        "name": "id",
                        "null_percentage": 0.0,
                        "statistics": {},
                    },
                    {
                        "name": "avg_rating",
                        "null_percentage": 8.04,
                        "statistics": {
                            "min": 2.8,
                            "max": 4.9,
                            "mean": 3.85,
                            "std": 0.61,
                        },
                    },
                    {
                        "name": "prep_capacity",
                        "null_percentage": 0.0,
                        "statistics": {
                            "min": -10.0,  # Negative anomaly
                            "max": 200.0,
                            "mean": 13.89,
                            "std": 24.81,
                        },
                    },
                ],
            }
        ]
    }

    # 3. Schema Registry
    schema_registry = {
        "datasets": {
            "restaurants": {
                "columns": {
                    "id": {"logical_dtype": "integer"},
                    "avg_rating": {"logical_dtype": "float"},
                    "prep_capacity": {"logical_dtype": "integer"},
                }
            }
        }
    }

    # 4. Feature Candidates
    feature_candidates = {
        "classifications": {
            "pii": [
                {
                    "dataset_id": "restaurants",
                    "column": "manager_contact",
                }
            ],
            "recommended_feature": [
                {
                    "dataset_id": "restaurants",
                    "column": "prep_capacity",
                    "encoding": "none",
                    "scaling": "standard_scaler",
                }
            ],
        }
    }

    # 5. Target Candidates
    target_candidates = {
        "strong_targets": [
            {
                "dataset_id": "orders",
                "column": "actual_delivery_time_min",
            }
        ]
    }

    # 6. Leakage Report
    leakage_report = {
        "findings": [
            {
                "dataset_id": "orders",
                "column": "promised_eta",
                "severity": "HIGH",
                "confidence": 0.98,
                "rationale": "Near-perfect correlation (0.999) with target.",
                "related_target": "actual_delivery_time_min",
            }
        ]
    }

    # Write files to mock directory
    (reports_dir / "validation_report.json").write_text(json.dumps(val_report))
    (reports_dir / "dataset_profile.json").write_text(json.dumps(dataset_profile))
    (reports_dir / "schema_registry.json").write_text(json.dumps(schema_registry))
    (reports_dir / "relationship_registry.json").write_text(json.dumps({}))
    (reports_dir / "feature_candidates.json").write_text(json.dumps(feature_candidates))
    (reports_dir / "target_candidates.json").write_text(json.dumps(target_candidates))
    (reports_dir / "leakage_report.json").write_text(json.dumps(leakage_report))
    (reports_dir / "intelligence_score.json").write_text(json.dumps({}))

    return reports_dir


def test_rule_engine(mock_reports_dir: Path) -> None:
    """Test that RuleEngine correctly parses input JSON files to find issues."""
    re = RuleEngine()
    reports = {
        "validation_report": json.loads((mock_reports_dir / "validation_report.json").read_text()),
        "dataset_profile": json.loads((mock_reports_dir / "dataset_profile.json").read_text()),
        "feature_candidates": json.loads((mock_reports_dir / "feature_candidates.json").read_text()),
        "leakage_report": json.loads((mock_reports_dir / "leakage_report.json").read_text()),
        "target_candidates": json.loads((mock_reports_dir / "target_candidates.json").read_text()),
    }
    issues = re.analyze(reports)

    types = {issue.issue_type for issue in issues}
    assert "schema_missing_columns" in types
    assert "null_violation" in types
    assert "duplicate_rows" in types
    assert "foreign_key_orphan" in types
    assert "negative_value_anomaly" in types
    assert "pii_column" in types
    assert "target_leakage" in types
    assert "numeric_needs_scaling" in types


def test_recommendation_engine() -> None:
    """Test that RecommendationEngine maps issues to correct cleaning actions."""
    re = RecommendationEngine()
    issues = [
        ActionableIssue(
            issue_type="duplicate_rows",
            dataset_id="restaurants",
            column=None,
            description="Duplicates found",
            severity=PriorityLevel.HIGH,
            source_report="test",
            details={"duplicate_row_count": 266},
        ),
        ActionableIssue(
            issue_type="pii_column",
            dataset_id="restaurants",
            column="manager_contact",
            description="PII",
            severity=PriorityLevel.HIGH,
            source_report="test",
        ),
        ActionableIssue(
            issue_type="target_leakage",
            dataset_id="orders",
            column="promised_eta",
            description="Leaks target",
            severity=PriorityLevel.HIGH,
            source_report="test",
        ),
    ]
    reports = {"schema_registry": {"datasets": {"restaurants": {"columns": {}}}}}
    recs = re.generate(issues, reports)

    actions = {rec.action for rec in recs}
    assert ActionVerb.REMOVE_DUPLICATES in actions
    assert ActionVerb.DROP in actions


def test_confidence_engine() -> None:
    """Test that ConfidenceEngine adjusts priority levels correctly."""
    engine = ConfidenceEngine()
    recs = [
        CleaningRecommendation(
            action=ActionVerb.IMPUTE,
            dataset_id="orders",
            column="actual_delivery_time_min",  # This is the prediction target
            confidence=0.90,
            priority=PriorityLevel.HIGH,
            rationale="Null values in target",
        )
    ]
    reports = {
        "target_candidates": {
            "strong_targets": [
                {
                    "dataset_id": "orders",
                    "column": "actual_delivery_time_min",
                }
            ]
        }
    }
    processed = engine.process(recs, reports)
    assert processed[0].priority == PriorityLevel.CRITICAL


def test_impact_estimator() -> None:
    """Test that ImpactEstimator calculates metrics properly."""
    estimator = ImpactEstimator()
    reports = {
        "validation_report": {
            "quality_score": 60.0,
            "component_scores": {"schema": 25.0, "nulls": 90.0, "duplicates": 95.0},
        },
        "leakage_report": {"findings": []},
        "feature_candidates": {"classifications": {"pii": []}},
    }
    recs = [
        CleaningRecommendation(
            action=ActionVerb.IMPUTE,
            dataset_id="restaurants",
            column="avg_rating",
            confidence=0.95,
            priority=PriorityLevel.HIGH,
            rationale="Impute rating",
        )
    ]
    impacts = estimator.estimate(reports, recs)
    assert impacts.data_quality.baseline == 60.0
    assert impacts.data_quality.target > 60.0
    assert impacts.completeness.delta > 0


def test_approval_manifest_builder() -> None:
    """Test that manifest entries have correct IDs and sorting."""
    builder = ApprovalManifestBuilder()
    recs = [
        CleaningRecommendation(
            action=ActionVerb.IMPUTE,
            dataset_id="restaurants",
            column="avg_rating",
            confidence=0.85,
            priority=PriorityLevel.MEDIUM,
            rationale="Impute",
        ),
        CleaningRecommendation(
            action=ActionVerb.DROP,
            dataset_id="restaurants",
            column="manager_contact",
            confidence=0.95,
            priority=PriorityLevel.HIGH,
            rationale="Drop PII",
        ),
    ]
    manifest = builder.build(recs)
    assert len(manifest) == 2
    assert manifest[0].decision_id == "DEC-001"
    assert manifest[0].priority == PriorityLevel.HIGH  # High priority sorted first
    assert manifest[1].decision_id == "DEC-002"


def test_report_generator_writes_reports(tmp_path: Path) -> None:
    """Test that DecisionReportGenerator writes all five reports."""
    generator = DecisionReportGenerator(tmp_path)
    recs = [
        CleaningRecommendation(
            action=ActionVerb.REMOVE_DUPLICATES,
            dataset_id="restaurants",
            column=None,
            confidence=1.0,
            priority=PriorityLevel.HIGH,
            rationale="Remove duplicate rows",
        )
    ]
    builder = ApprovalManifestBuilder()
    manifest = builder.build(recs)
    estimator = ImpactEstimator()
    reports = {
        "validation_report": {"quality_score": 60.0},
        "leakage_report": {"findings": []},
        "feature_candidates": {"classifications": {}},
    }
    impacts = estimator.estimate(reports, recs)

    paths = generator.generate(recs, manifest, impacts, 1)

    assert "cleaning_plan_json" in paths
    assert "cleaning_plan_md" in paths
    assert "decision_summary_json" in paths
    assert "estimated_quality_json" in paths
    assert "approval_manifest_json" in paths

    assert (tmp_path / "cleaning_plan.json").exists()
    assert (tmp_path / "cleaning_plan.md").exists()
    assert (tmp_path / "decision_summary.json").exists()
    assert (tmp_path / "estimated_quality.json").exists()
    assert (tmp_path / "approval_manifest.json").exists()


def test_decision_engine_e2e(mock_reports_dir: Path) -> None:
    """Test that DecisionEngine coordinates and runs successfully."""
    config = DecisionConfig(reports_dir=mock_reports_dir)
    engine = DecisionEngine(config=config)
    paths = engine.run()

    assert "cleaning_plan_json" in paths
    assert "approval_manifest_json" in paths
    assert paths["cleaning_plan_json"].exists()

    # Verify contents of generated manifests
    manifest_data = json.loads(paths["approval_manifest_json"].read_text())
    assert "manifest" in manifest_data
    assert len(manifest_data["manifest"]) > 0


def test_cli_execution(mock_reports_dir: Path) -> None:
    """Test the CLI entry point execution with mock reports."""
    exit_code = main(["--reports-dir", str(mock_reports_dir)])
    assert exit_code == 0
