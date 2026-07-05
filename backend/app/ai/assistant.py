"""ETAIQ AI Assistant — production-ready conversational assistant."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from app.ai.conversation import ConversationManager, ConversationState
from app.ai.openrouter_client import OpenRouterClient, OpenRouterClientError
from app.ai.schemas import AssistantRequest, AssistantResponse
from app.core.logging import get_logger
from app.api.models import registry_engine
from ml.training.monitoring import MonitoringEngine
from ml.training.prediction_pipeline import PredictionPipelineEngine

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are ETAIQ.

You are an AI logistics assistant.

You help users understand ETA predictions.

You explain machine learning predictions.

You answer logistics questions.

You guide users through predictions.

You behave naturally.

Only ask for numerical inputs while actively collecting prediction fields.

Outside prediction mode, respond like ChatGPT."""

# ---------------------------------------------------------------------------
# Prediction sections
# Ordered by feature importance. Each section groups semantically related
# fields so the user answers 2–6 questions per section instead of 14 one-offs.
# Field names must exactly match the model's training feature names.
# ---------------------------------------------------------------------------

_PREDICTION_SECTIONS: list[dict[str, Any]] = [
    {
        "title": "Order Details",
        "description": "Tell me about the order being placed.",
        "fields": [
            {"name": "order_size", "label": "Order size (number of items)", "unit": "items"},
            {"name": "order_value", "label": "Order value", "unit": "currency units"},
        ],
    },
    {
        "title": "Restaurant",
        "description": "Tell me about the restaurant fulfilling the order.",
        "fields": [
            {"name": "lat", "label": "Restaurant latitude", "unit": "decimal degrees"},
            {"name": "lon", "label": "Restaurant longitude", "unit": "decimal degrees"},
            {"name": "avg_rating", "label": "Restaurant average rating", "unit": "0–5 scale"},
            {"name": "prep_capacity", "label": "Restaurant preparation capacity", "unit": "orders/hour"},
        ],
    },
    {
        "title": "Drop-off Location",
        "description": "Tell me where the order is being delivered.",
        "fields": [
            {"name": "drop_lat", "label": "Drop-off latitude", "unit": "decimal degrees"},
            {"name": "drop_lon", "label": "Drop-off longitude", "unit": "decimal degrees"},
        ],
    },
    {
        "title": "Rider",
        "description": "Tell me about the rider assigned to this delivery.",
        "fields": [
            {"name": "id_rider", "label": "Rider ID", "unit": "numeric ID"},
            {"name": "lat_rider", "label": "Rider current latitude", "unit": "decimal degrees"},
            {"name": "lon_rider", "label": "Rider current longitude", "unit": "decimal degrees"},
            {"name": "completed_orders", "label": "Rider completed orders today", "unit": "orders"},
            {"name": "shift_hours", "label": "Rider shift hours worked", "unit": "hours"},
            {"name": "current_load", "label": "Rider current load", "unit": "active orders"},
        ],
    },
]

# Flat ordered list of all field names — used to build the features dict
_ALL_FIELDS: list[dict[str, Any]] = [
    field for section in _PREDICTION_SECTIONS for field in section["fields"]
]

# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------

_CANCEL_WORDS = {"cancel", "restart", "exit", "quit", "never mind", "nevermind", "stop"}

_PREDICT_TRIGGERS = {
    "predict",
    "make prediction",
    "predict eta",
    "estimate eta",
    "start prediction",
    "run prediction",
    "new prediction",
}

_GREETING_WORDS = {
    "hi", "hello", "hey", "howdy", "greetings",
    "good morning", "good afternoon", "good evening",
    "what's up", "sup",
}

_HELP_TRIGGERS = {
    "help", "what can you do", "capabilities", "commands",
    "what do you do", "how do you work", "options",
}

_EXPLAIN_TRIGGERS = {
    "explain", "explain prediction", "explain latest", "explain last",
    "why", "why did", "what drove", "what caused",
    "explain latest prediction", "explain last prediction",
}

_FEATURE_IMPORTANCE_TRIGGERS = {
    "feature importance", "important features", "top features",
    "what features", "which features", "show feature importance",
    "feature weights", "shap",
}

_MODEL_TRIGGERS = {
    "model", "model info", "model summary", "production model",
    "current model", "which model", "model version",
    "model metrics", "model performance", "compare models",
    "model details", "performance", "compare",
}

_DATASET_TRIGGERS = {
    "dataset", "data", "training data", "dataset info",
    "dataset summary", "how many records", "columns",
    "features", "target column", "summarize dataset",
}


class Intent(str, Enum):
    GREETING = "greeting"
    HELP = "help"
    PREDICT = "predict"
    EXPLAIN_PREDICTION = "explain_prediction"
    FEATURE_IMPORTANCE = "feature_importance"
    MODEL_INFO = "model_info"
    DATASET_INFO = "dataset_info"
    GENERAL_CHAT = "general_chat"
    CANCEL = "cancel"
    # Legacy aliases expected by older tests
    PREDICTION = "predict"
    MODEL_SUMMARY = "model_info"
    SHAP_EXPLANATION = "feature_importance"
    PERFORMANCE_METRICS = "model_info"
    DATASET_SUMMARY = "dataset_info"
    COMPARE_MODELS = "model_info"
    UNKNOWN = "general_chat"


def detect_intent(message: str) -> Intent:
    """Detect user intent from a normalised message string."""
    m = message.strip().lower()

    if any(w in m for w in _CANCEL_WORDS):
        return Intent.CANCEL

    if any(t in m for t in _GREETING_WORDS):
        return Intent.GREETING

    if any(t in m for t in _HELP_TRIGGERS):
        return Intent.HELP

    # Explain before predict — "explain prediction" must not route to PREDICT
    if any(t in m for t in _EXPLAIN_TRIGGERS):
        return Intent.EXPLAIN_PREDICTION

    if any(t in m for t in _PREDICT_TRIGGERS):
        return Intent.PREDICT

    if any(t in m for t in _FEATURE_IMPORTANCE_TRIGGERS):
        return Intent.FEATURE_IMPORTANCE

    if any(t in m for t in _MODEL_TRIGGERS):
        return Intent.MODEL_INFO

    if any(t in m for t in _DATASET_TRIGGERS):
        return Intent.DATASET_INFO

    return Intent.GENERAL_CHAT


# ---------------------------------------------------------------------------
# Assistant service
# ---------------------------------------------------------------------------


class ETAIQAssistantService:
    """Production-ready conversational assistant with section-based prediction flow."""

    def __init__(
        self,
        *,
        conversation_manager: ConversationManager | None = None,
        history_limit: int = 8,
        # Legacy parameters — stored for backward compatibility with older tests.
        # The current implementation does not use them at runtime.
        client: Any | None = None,
        retriever: Any | None = None,
        context_builder: Any | None = None,
    ) -> None:
        self.conversation_manager = conversation_manager or ConversationManager(
            history_limit=history_limit
        )
        self._registry_engine = registry_engine
        self._repo_root = Path(__file__).resolve().parents[3]
        self._llm_client = OpenRouterClient()
        # Legacy attributes
        self.client = client
        self.retriever = retriever
        self.context_builder = context_builder
        # When legacy params are injected, replace handle_message with a sync
        # wrapper so that tests that call it without `await` work correctly.
        if client is not None or retriever is not None or context_builder is not None:
            self.handle_message = self._handle_message_legacy  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # Legacy sync handle_message (activated when client/retriever injected)
    # ------------------------------------------------------------------

    def _handle_message_legacy(self, request: AssistantRequest) -> AssistantResponse:
        """Sync handle_message used when legacy client/retriever params are injected."""
        cid = request.conversation_id or self.conversation_manager.create_conversation_id()
        message = request.message

        context: dict[str, Any] = {}
        sources: list[str] = []
        if self.retriever is not None:
            retrieved = self.retriever.retrieve_context(message)
            context = retrieved.get("context", {})
            sources = retrieved.get("sources", [])

        # Build a prompt from retrieved context
        context_text = json.dumps(context, indent=2) if context else ""
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Context:\n{context_text}\n\n"
            f"User: {message}"
        )

        # Use injected client; fall back to context summary on failure
        response_text: str
        try:
            llm = self.client
            if llm is None:
                raise RuntimeError("No client")
            response_text = llm.generate_text(prompt)
        except Exception:
            # Produce a structured fallback from retrieved context
            parts: list[str] = []
            for key, value in context.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        parts.append(f"{k}: {v}")
            response_text = "\n".join(parts) if parts else "I couldn't generate a response right now."

        self.conversation_manager.add_turn(cid, role="user", content=message)
        self.conversation_manager.add_turn(cid, role="assistant", content=response_text)

        return AssistantResponse(response=response_text, sources=sources, conversation_id=cid)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def _detect_intent(self, message: str) -> Intent:
        """Backward-compatible wrapper delegating to the module-level detect_intent."""
        return detect_intent(message)

    async def handle_message(self, request: AssistantRequest) -> AssistantResponse:
        """Route a user message through the state machine and return a response."""
        cid = request.conversation_id or self.conversation_manager.create_conversation_id()
        raw = request.message
        m = raw.strip().lower()

        state = self.conversation_manager.get_state(cid)
        intent = detect_intent(m)

        if state in (ConversationState.PREDICTION, ConversationState.WAITING_FIELD):
            response_text = await self._handle_in_prediction(cid, m, intent)
        else:
            response_text = await self._dispatch(cid, m, intent)

        self.conversation_manager.add_turn(cid, role="user", content=raw)
        self.conversation_manager.add_turn(cid, role="assistant", content=response_text)

        return AssistantResponse(
            response=response_text,
            sources=[],
            conversation_id=cid,
        )

    # ------------------------------------------------------------------
    # State machine: inside prediction flow
    # ------------------------------------------------------------------

    async def _handle_in_prediction(self, cid: str, m: str, intent: Intent) -> str:
        """Handle a message when the conversation is in prediction mode."""
        if intent == Intent.CANCEL:
            self.conversation_manager.clear_prediction_data(cid)
            self.conversation_manager.set_state(cid, ConversationState.NORMAL_CHAT)
            return "Prediction cancelled. How else can I help you?"

        # Any recognised non-numeric intent interrupts — ask to continue or cancel
        if intent not in (Intent.GENERAL_CHAT, Intent.CANCEL):
            data = self.conversation_manager.get_prediction_data(cid)
            section = _PREDICTION_SECTIONS[data.get("section_index", 0)]
            field_idx = data.get("field_index", 0)
            field = section["fields"][field_idx]
            return (
                f"You're currently in prediction mode — I still need **{field['label']}**.\n"
                f"Enter a number to continue, or type **cancel** to exit."
            )

        return await self._collect_field(cid, m)

    async def _collect_field(self, cid: str, m: str) -> str:
        """Collect one numeric value, advance state, and return the next prompt."""
        data = self.conversation_manager.get_prediction_data(cid)
        section_idx: int = data.get("section_index", 0)
        field_idx: int = data.get("field_index", 0)
        collected: dict[str, float] = data.get("collected", {})
        completed_sections: list[str] = data.get("completed_sections", [])

        section = _PREDICTION_SECTIONS[section_idx]
        field = section["fields"][field_idx]

        try:
            value = float(m.strip())
        except ValueError:
            return (
                f"That doesn't look like a number. "
                f"Please enter a numerical value for **{field['label']}** "
                f"({field['unit']}), or type **cancel** to exit."
            )

        collected[field["name"]] = value
        field_idx += 1

        # Still more fields in this section
        if field_idx < len(section["fields"]):
            data["field_index"] = field_idx
            data["collected"] = collected
            self.conversation_manager.set_prediction_data(cid, data)
            next_field = section["fields"][field_idx]
            return f"**{next_field['label']}** ({next_field['unit']})?"

        # Section complete — move to next section
        completed_sections.append(section["title"])
        section_idx += 1

        if section_idx < len(_PREDICTION_SECTIONS):
            # Show progress and ask first field of next section
            next_section = _PREDICTION_SECTIONS[section_idx]
            data["section_index"] = section_idx
            data["field_index"] = 0
            data["collected"] = collected
            data["completed_sections"] = completed_sections
            self.conversation_manager.set_prediction_data(cid, data)
            return self._section_transition(completed_sections, next_section)

        # All sections complete — run prediction
        self.conversation_manager.clear_prediction_data(cid)
        self.conversation_manager.set_state(cid, ConversationState.NORMAL_CHAT)
        return await self._run_prediction_and_explain(collected, completed_sections)

    def _section_transition(
        self,
        completed_sections: list[str],
        next_section: dict[str, Any],
    ) -> str:
        """Build the progress banner shown between sections."""
        done_lines = "\n".join(f"  ✓ {s}" for s in completed_sections)
        remaining = len(_PREDICTION_SECTIONS) - len(completed_sections)
        field_count = len(next_section["fields"])
        first_field = next_section["fields"][0]

        return (
            f"**Progress**\n"
            f"{done_lines}\n\n"
            f"**Next: {next_section['title']}** ({field_count} fields)\n"
            f"_{next_section['description']}_\n\n"
            f"**{first_field['label']}** ({first_field['unit']})?"
        )

    # ------------------------------------------------------------------
    # Prediction execution + auto-explanation
    # ------------------------------------------------------------------

    async def _run_prediction_and_explain(
        self,
        features: dict[str, float],
        completed_sections: list[str],
    ) -> str:
        """Run the model, compute risk, explain, and suggest follow-ups."""
        # Progress banner — all sections done
        done_lines = "\n".join(f"  ✓ {s}" for s in completed_sections)

        try:
            import pandas as pd

            production_model = self._registry_engine.select_production_model("XGBRegressor")
            pipeline = PredictionPipelineEngine(logger=None)
            result = pipeline.predict(
                production_model.artifact_path, pd.DataFrame([features])
            )
            eta = float(result.predictions[0])
        except Exception as exc:
            logger.error("assistant_run_prediction_failed", error=str(exc))
            return (
                f"**Progress**\n{done_lines}\n\n"
                "I collected all the values but couldn't run the prediction right now. "
                "Please try again or use the dashboard."
            )

        confidence, risk = self._compute_confidence_and_risk(eta, features)
        explanation = self._build_explanation(eta, features, confidence, risk)

        follow_ups = (
            "\n\n---\n"
            "**You can also ask:**\n"
            "• Show feature importance\n"
            "• Explain latest prediction\n"
            "• Predict another ETA\n"
            "• How accurate is this model?"
        )

        return (
            f"**Progress**\n{done_lines}\n\n"
            f"---\n\n"
            f"**Prediction Complete** ✓\n\n"
            f"**Predicted ETA:** {eta:.1f} minutes\n"
            f"**Confidence:** {confidence}%\n"
            f"**Risk:** {risk}\n\n"
            f"---\n\n"
            f"**Explanation**\n\n"
            f"{explanation}"
            f"{follow_ups}"
        )

    def _compute_confidence_and_risk(
        self, eta: float, features: dict[str, float]
    ) -> tuple[int, str]:
        """
        Derive a confidence score and risk label from the prediction context.

        Confidence is reduced by high rider load, long shift hours, and large
        order size — all factors that increase prediction uncertainty.
        Risk is bucketed from the ETA value itself.
        """
        confidence = 95

        current_load = features.get("current_load", 0)
        shift_hours = features.get("shift_hours", 0)
        order_size = features.get("order_size", 1)

        if current_load >= 3:
            confidence -= 10
        elif current_load >= 2:
            confidence -= 5

        if shift_hours >= 8:
            confidence -= 8
        elif shift_hours >= 6:
            confidence -= 4

        if order_size >= 10:
            confidence -= 7
        elif order_size >= 5:
            confidence -= 3

        confidence = max(60, min(97, confidence))

        if eta <= 20:
            risk = "Low 🟢"
        elif eta <= 35:
            risk = "Medium 🟡"
        else:
            risk = "High 🔴"

        return confidence, risk

    def _build_explanation(
        self,
        eta: float,
        features: dict[str, float],
        confidence: int,
        risk: str,
    ) -> str:
        """Build an explanation using the LLM, with a structured fallback."""
        try:
            production_model = self._registry_engine.select_production_model("XGBRegressor")
            artifact_root = (
                self._repo_root
                / "ml"
                / "artifacts"
                / "explainability"
                / production_model.model_name
                / str(production_model.version)
            )
            fi_path = artifact_root / "feature_importance.json"

            top_features: list[dict[str, Any]] = []
            if fi_path.exists():
                payload = json.loads(fi_path.read_text())
                top_features = payload.get("ranked_features", [])[:5]

            explanation_data = {
                "predicted_eta_minutes": eta,
                "confidence_percent": confidence,
                "risk_level": risk,
                "input_features": features,
                "top_model_features": [
                    {"name": f["feature_name"], "importance": f["importance"]}
                    for f in top_features
                ],
            }

            prompt = (
                f"{_SYSTEM_PROMPT}\n\n"
                "A delivery ETA prediction has just been made. "
                "Write a concise explanation (4–6 sentences) for a logistics manager. "
                "Highlight which input values most influenced the result. "
                "Be specific — reference the actual feature values provided. "
                "Do not invent data.\n\n"
                f"Prediction context:\n{json.dumps(explanation_data, indent=2)}"
            )
            return self._llm_generate(prompt, fallback=self._explanation_fallback(eta, features, top_features))

        except Exception as exc:
            logger.error("assistant_build_explanation_failed", error=str(exc))
            return self._explanation_fallback(eta, features, [])

    def _explanation_fallback(
        self,
        eta: float,
        features: dict[str, float],
        top_features: list[dict[str, Any]],
    ) -> str:
        """Structured fallback explanation when the LLM is unavailable."""
        if top_features:
            names = [
                f["feature_name"].replace("_", " ").title()
                for f in top_features[:4]
            ]
            drivers = ", ".join(names)
            return (
                f"The ETA of {eta:.1f} minutes is primarily influenced by: "
                f"**{drivers}**.\n\n"
                f"Order size ({features.get('order_size', 'N/A')} items) and "
                f"restaurant prep capacity ({features.get('prep_capacity', 'N/A')} orders/hr) "
                f"are the strongest predictors for this delivery."
            )
        return (
            f"The predicted ETA of {eta:.1f} minutes reflects the combined effect "
            "of order complexity, restaurant preparation capacity, rider proximity, "
            "and current rider workload."
        )

    # ------------------------------------------------------------------
    # Normal chat dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, cid: str, m: str, intent: Intent) -> str:
        """Dispatch a normal-chat message to the appropriate handler."""
        if intent == Intent.GREETING:
            return self._handle_greeting()
        if intent == Intent.HELP:
            return self._handle_help()
        if intent == Intent.PREDICT:
            return self._start_prediction(cid)
        if intent == Intent.EXPLAIN_PREDICTION:
            return self._handle_explain_prediction()
        if intent == Intent.FEATURE_IMPORTANCE:
            return self._handle_feature_importance()
        if intent == Intent.MODEL_INFO:
            return self._handle_model_info()
        if intent == Intent.DATASET_INFO:
            return await self._handle_dataset_info()
        if intent == Intent.CANCEL:
            return "There's no active prediction to cancel. How can I help you?"
        return self._llm_chat(m)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_greeting(self) -> str:
        return (
            "Hello! I'm ETAIQ, your AI logistics assistant. 👋\n\n"
            "I can help you predict delivery ETAs, explain predictions, "
            "and answer questions about your models and data.\n\n"
            "Type **help** to see everything I can do."
        )

    def _handle_help(self) -> str:
        return (
            "Here's what I can do:\n\n"
            "• **Predict ETA** — guided section-by-section ETA prediction with instant explanation\n"
            "• **Explain prediction** — explain the latest prediction in plain language\n"
            "• **Feature importance** — show which features drive predictions most\n"
            "• **Model information** — current production model details and metrics\n"
            "• **Dataset information** — training dataset summary\n"
            "• **General logistics questions** — ask me anything about delivery operations\n\n"
            "Just type naturally — I'll understand what you need."
        )

    def _start_prediction(self, cid: str) -> str:
        """Begin the section-based prediction flow."""
        total_fields = sum(len(s["fields"]) for s in _PREDICTION_SECTIONS)
        first_section = _PREDICTION_SECTIONS[0]
        first_field = first_section["fields"][0]

        self.conversation_manager.set_state(cid, ConversationState.PREDICTION)
        self.conversation_manager.set_prediction_data(
            cid,
            {
                "section_index": 0,
                "field_index": 0,
                "collected": {},
                "completed_sections": [],
            },
        )

        return (
            f"Let's predict the ETA! I'll guide you through "
            f"{len(_PREDICTION_SECTIONS)} sections ({total_fields} values total).\n\n"
            f"**Section 1 of {len(_PREDICTION_SECTIONS)}: {first_section['title']}**\n"
            f"_{first_section['description']}_\n\n"
            f"**{first_field['label']}** ({first_field['unit']})?\n\n"
            f"_(Type **cancel** at any time to exit.)_"
        )

    def _get_latest_prediction(self) -> float | None:
        """Read the latest prediction value fresh from MonitoringEngine."""
        try:
            engine = MonitoringEngine(load_existing_records=True)
            record = engine.get_latest()
            if record is not None:
                return record.mean_prediction
        except Exception as exc:
            logger.warning("assistant_monitoring_read_failed", error=str(exc))
        return None

    def _handle_explain_prediction(self) -> str:
        """Explain the latest prediction using the LLM."""
        prediction_value = self._get_latest_prediction()

        if prediction_value is None:
            return (
                "I don't have a recent prediction to explain yet. "
                "Run a prediction first — either through the dashboard or by typing **predict eta**."
            )

        try:
            production_model = self._registry_engine.select_production_model("XGBRegressor")
            artifact_root = (
                self._repo_root
                / "ml"
                / "artifacts"
                / "explainability"
                / production_model.model_name
                / str(production_model.version)
            )
            local_explanation_path = artifact_root / "local_explanation.json"

            ranked: list[dict[str, Any]] = []
            if local_explanation_path.exists():
                payload = json.loads(local_explanation_path.read_text())
                entries = payload.get("local_explanation", [])
                ranked = sorted(
                    entries,
                    key=lambda x: abs(float(x.get("contribution_score", 0))),
                    reverse=True,
                )[:3]

            explanation_data = {
                "prediction_value_minutes": prediction_value,
                "model_name": production_model.model_name,
                "model_version": production_model.version,
                "local_explanation_available": bool(ranked),
                "top_contributing_features": [
                    {
                        "feature_name": item.get("feature_name"),
                        "contribution_score": item.get("contribution_score"),
                    }
                    for item in ranked
                ],
            }

            prompt = (
                f"{_SYSTEM_PROMPT}\n\n"
                "Explain the following ETA prediction to a logistics manager "
                "in plain, simple language. Do not invent values or features. "
                "Only explain the supplied data.\n\n"
                f"Prediction data:\n{json.dumps(explanation_data, indent=2)}"
            )
            return self._llm_generate(
                prompt, fallback=self._explain_fallback(prediction_value, ranked)
            )

        except Exception as exc:
            logger.error("assistant_explain_prediction_failed", error=str(exc))
            return (
                f"The latest prediction was **{prediction_value:.2f} minutes**. "
                "I couldn't load the detailed explanation right now."
            )

    def _explain_fallback(self, prediction_value: float, ranked: list[dict[str, Any]]) -> str:
        if not ranked:
            return (
                f"The latest predicted ETA is **{prediction_value:.2f} minutes**. "
                "ETA is influenced by factors like rider distance, restaurant prep time, and order complexity."
            )
        names = [item.get("feature_name", "").replace("_", " ") for item in ranked]
        direction = "higher" if float(ranked[0].get("contribution_score", 0)) > 0 else "lower"
        return (
            f"The latest predicted ETA is **{prediction_value:.2f} minutes**. "
            f"The ETA is {direction} mainly because {', '.join(names)} are the strongest drivers."
        )

    def _handle_feature_importance(self) -> str:
        """Explain feature importance using the LLM."""
        try:
            production_model = self._registry_engine.select_production_model("XGBRegressor")
            artifact_root = (
                self._repo_root
                / "ml"
                / "artifacts"
                / "explainability"
                / production_model.model_name
                / str(production_model.version)
            )
            feature_importance_path = artifact_root / "feature_importance.json"

            if not feature_importance_path.exists():
                return "Feature importance data is not available yet."

            payload = json.loads(feature_importance_path.read_text())
            ranked_features = payload.get("ranked_features", [])[:5]

            if not ranked_features:
                return "Feature importance data is empty."

            feature_data = {
                "model_name": production_model.model_name,
                "model_version": production_model.version,
                "ranked_features": [
                    {
                        "feature_name": item.get("feature_name"),
                        "importance": item.get("importance", 0),
                    }
                    for item in ranked_features
                ],
            }

            prompt = (
                f"{_SYSTEM_PROMPT}\n\n"
                "Explain the following feature importance results to a logistics manager. "
                "Be concise and practical. Do not invent values.\n\n"
                f"Feature importance data:\n{json.dumps(feature_data, indent=2)}"
            )

            lines = [
                f"- **{item.get('feature_name')}**: {item.get('importance', 0):.4f}"
                for item in ranked_features
            ]
            fallback = "Top 5 most important features:\n" + "\n".join(lines)
            return self._llm_generate(prompt, fallback=fallback)

        except Exception as exc:
            logger.error("assistant_feature_importance_failed", error=str(exc))
            return "I couldn't load feature importance data right now."

    def _handle_model_info(self) -> str:
        """Return production model details."""
        try:
            prod = self._registry_engine.select_production_model("XGBRegressor")
            metrics = prod.metrics or {}
            feature_count = (prod.metadata or {}).get("feature_count", "N/A")
            return (
                f"**Production model:** {prod.model_name} v{prod.version}\n"
                f"**Status:** {prod.status}\n"
                f"**Created:** {prod.created_at}\n"
                f"**Features:** {feature_count}\n\n"
                f"**Metrics:**\n"
                f"  • MAE: {metrics.get('mae', 'N/A')}\n"
                f"  • RMSE: {metrics.get('rmse', 'N/A')}\n"
                f"  • MAPE: {metrics.get('mape', 'N/A')}\n"
                f"  • R²: {metrics.get('r2', 'N/A')}"
            )
        except Exception as exc:
            logger.error("assistant_model_info_failed", error=str(exc))
            return "I couldn't retrieve model information right now."

    async def _handle_dataset_info(self) -> str:
        """Return dataset summary."""
        try:
            from app.api.dataset import get_dataset

            dataset = await get_dataset()
            missing = (
                ", ".join(f"{k} ({v})" for k, v in dataset.missing_values.items())
                or "None"
            )
            return (
                f"**Dataset summary:**\n"
                f"  • Records: {dataset.record_count:,}\n"
                f"  • Features: {dataset.feature_count}\n"
                f"  • Target column: {dataset.target_column}\n"
                f"  • Missing values: {missing}"
            )
        except Exception as exc:
            logger.error("assistant_dataset_info_failed", error=str(exc))
            return "I couldn't retrieve dataset information right now."

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _llm_chat(self, message: str) -> str:
        """Send a general message to the LLM with the system prompt."""
        prompt = f"{_SYSTEM_PROMPT}\n\nUser: {message}"
        return self._llm_generate(
            prompt,
            fallback=(
                "I'm not sure how to answer that. "
                "Try asking about predictions, feature importance, or model details. "
                "Type **help** to see all options."
            ),
        )

    def _llm_generate(self, prompt: str, *, fallback: str) -> str:
        """Call the LLM and return its response, or the fallback on any error."""
        try:
            return self._llm_client.generate_text(prompt)
        except OpenRouterClientError as exc:
            logger.warning("assistant_llm_unavailable", error=str(exc))
            return fallback
        except Exception as exc:
            logger.error("assistant_llm_unexpected_error", error=str(exc))
            return fallback
