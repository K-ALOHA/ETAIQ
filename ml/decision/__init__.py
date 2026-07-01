"""ETAIQ Decision Intelligence Engine for generating explainable cleaning plans."""

__all__ = ["DecisionEngine", "run_decision"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from ml.decision.decision_engine import DecisionEngine
        from ml.decision.main import run_decision

        return {
            "DecisionEngine": DecisionEngine,
            "run_decision": run_decision,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
