"""Prompt construction helpers for the ETAIQ AI assistant."""

from __future__ import annotations

from typing import Any


def build_system_prompt() -> str:
    """Return the production system prompt for the ETAIQ assistant."""
    return (
        "You are ETAIQ Assistant, an AI support assistant for the ETAIQ platform. "
        "Answer only about ETAIQ, machine learning, ETA prediction, datasets, monitoring, "
        "explainability, model registry, training, and platform usage. "
        "If the user asks an unrelated question, politely redirect them to ETAIQ topics. "
        "Be concise, practical, and grounded in the ETAIQ platform."
    )


def build_prompt(
    *,
    system_prompt: str,
    context: dict[str, Any],
    history: list[dict[str, str]],
    user_message: str,
    is_related: bool,
) -> str:
    """Compose the final prompt sent to the LLM."""
    history_section = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in history
    ) if history else "No prior conversation history."

    context_section = "\n".join(
        f"{key}: {value}" for key, value in context.items()
    ) if context else "No additional context provided."

    redirect_instruction = ""
    if not is_related:
        redirect_instruction = (
            "The user request is unrelated to ETAIQ. Respond by politely redirecting "
            "them to ETAIQ, ETA prediction, monitoring, explainability, training, or platform usage."
        )

    return (
        f"{system_prompt}\n\n"
        f"{redirect_instruction}\n\n"
        f"Context:\n{context_section}\n\n"
        f"Conversation history:\n{history_section}\n\n"
        f"User message:\n{user_message}"
    )
