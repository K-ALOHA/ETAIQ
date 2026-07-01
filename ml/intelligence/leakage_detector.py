"""Target leakage detection with confidence and actionable recommendations."""

from __future__ import annotations

import time

import pandas as pd

from ml.intelligence.config import (
    DEFAULT_CONFIG,
    POST_EVENT_LEAKAGE_PATTERN,
    IntelligenceConfig,
)
from ml.intelligence.logging_config import get_logger
from ml.intelligence.models import (
    ColumnProfile,
    DatasetProfile,
    LeakageFinding,
    TargetCandidate,
)

logger = get_logger(__name__)


class LeakageDetector:
    """Detects columns that may leak target information."""

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
        targets: list[TargetCandidate],
    ) -> list[LeakageFinding]:
        """Find leakage risks relative to high-confidence target candidates."""
        logger.info("leakage_detection_start", targets=len(targets))
        start = time.perf_counter()
        findings: list[LeakageFinding] = []

        strong_targets = [t for t in targets if t.tier in {"strong", "possible"}]
        if not strong_targets:
            strong_targets = targets[:3]

        profile_map = {profile.dataset_id: profile for profile in profiles}
        for target in strong_targets:
            frame = frames.get(target.dataset_id)
            profile = profile_map.get(target.dataset_id)
            if frame is None or profile is None or target.column not in frame.columns:
                continue

            target_series = pd.to_numeric(frame[target.column], errors="coerce")
            for column_profile in profile.columns:
                if column_profile.name == target.column:
                    continue
                finding = self._evaluate_column(
                    target.dataset_id,
                    column_profile,
                    frame[column_profile.name],
                    target_series,
                    target.column,
                )
                if finding:
                    findings.append(finding)

        findings.sort(key=lambda item: item.confidence, reverse=True)
        logger.info(
            "leakage_detection_end",
            findings=len(findings),
            duration_seconds=round(time.perf_counter() - start, 4),
        )
        return findings

    def _evaluate_column(
        self,
        dataset_id: str,
        column_profile: ColumnProfile,
        series: pd.Series,
        target_series: pd.Series,
        target_name: str,
    ) -> LeakageFinding | None:
        """Evaluate a single column for leakage against a target."""
        name = column_profile.name

        if POST_EVENT_LEAKAGE_PATTERN.search(name) and (
            "timestamp" in column_profile.roles or column_profile.is_datetime
        ):
            return LeakageFinding(
                dataset_id=dataset_id,
                column=name,
                severity="HIGH",
                confidence=0.95,
                rationale="Available after prediction event.",
                recommendation="Do not use. Post-event timestamp leaks outcome timing.",
                related_target=target_name,
            )

        if target_name.lower() in name.lower() and name.lower() != target_name.lower():
            return LeakageFinding(
                dataset_id=dataset_id,
                column=name,
                severity="HIGH",
                confidence=0.92,
                rationale="Column name appears derived from the target variable.",
                recommendation="Do not use. Likely a transformed target value.",
                related_target=target_name,
            )

        numeric = pd.to_numeric(series, errors="coerce")
        aligned = pd.concat([numeric, target_series], axis=1).dropna()
        if len(aligned) >= self._config.leakage_min_correlation_samples:
            correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            if (
                correlation is not None
                and abs(correlation) >= self._config.leakage_correlation_high
            ):
                return LeakageFinding(
                    dataset_id=dataset_id,
                    column=name,
                    severity="HIGH",
                    confidence=0.98,
                    rationale=f"Near-perfect correlation ({correlation:.3f}) with target.",
                    recommendation="Do not use. Direct proxy for the target.",
                    related_target=target_name,
                )
            if (
                correlation is not None
                and abs(correlation) >= self._config.leakage_correlation_medium
            ):
                return LeakageFinding(
                    dataset_id=dataset_id,
                    column=name,
                    severity="MEDIUM",
                    confidence=0.85,
                    rationale=f"High correlation ({correlation:.3f}) with target.",
                    recommendation="Review carefully; may encode target information.",
                    related_target=target_name,
                )
            if (
                correlation is not None
                and abs(correlation) >= self._config.leakage_correlation_low
            ):
                return LeakageFinding(
                    dataset_id=dataset_id,
                    column=name,
                    severity="LOW",
                    confidence=0.65,
                    rationale=f"Moderate correlation ({correlation:.3f}) with target.",
                    recommendation="Monitor during feature selection.",
                    related_target=target_name,
                )

        return None

    @staticmethod
    def annotate_profiles(
        profiles: list[DatasetProfile],
        findings: list[LeakageFinding],
    ) -> None:
        """Mark columns with potential leakage roles."""
        finding_columns = {(item.dataset_id, item.column) for item in findings}
        for profile in profiles:
            for column in profile.columns:
                if (profile.dataset_id, column.name) in finding_columns:
                    if "potential_leakage" not in column.roles:
                        column.roles.append("potential_leakage")
