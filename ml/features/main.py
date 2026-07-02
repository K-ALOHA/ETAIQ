"""Main entry point for the feature engineering module."""

from __future__ import annotations

from .config import DEFAULT_FEATURE_ENGINEERING_CONFIG
from .feature_engineering import FeatureEngineeringEngine
from .feature_pipeline import FeaturePipeline
from .feature_registry import FeatureRegistryManager
from .logging_config import FeatureEngineeringLogger
from .utils import print_banner


def main() -> None:
    """Initialize the feature engineering architecture and print readiness status."""
    print_banner("ETAIQ Feature Engineering Module")

    config = DEFAULT_FEATURE_ENGINEERING_CONFIG
    logger = FeatureEngineeringLogger(level=config.log_level)
    registry = FeatureRegistryManager(config=config, logger=logger)
    pipeline = FeaturePipeline(config=config, logger=logger, registry=registry)
    engine = FeatureEngineeringEngine(config=config, logger=logger)

    registry.initialize()
    pipeline.run()
    engine.run()

    print("Configuration Loaded")
    print("Logger Ready")
    print("Registry Ready")
    print("Pipeline Ready")
    print("Status : READY")


if __name__ == "__main__":
    main()
