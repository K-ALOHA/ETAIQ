"""Dataset Intelligence Engine for automatic dataset discovery and profiling."""

__all__ = ["IntelligenceEngine", "run_intelligence"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from ml.intelligence.main import IntelligenceEngine, run_intelligence

        return {
            "IntelligenceEngine": IntelligenceEngine,
            "run_intelligence": run_intelligence,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
