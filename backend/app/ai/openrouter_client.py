"""OpenRouter client wrapper for the ETAIQ AI assistant."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenRouterClientError(RuntimeError):
    """Raised when the OpenRouter client cannot produce a response."""


class OpenRouterClient:
    """Reusable wrapper around the OpenRouter API via the OpenAI SDK."""

    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or settings.openrouter_model or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3")
        self.base_url = settings.openrouter_base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))

        self._client: OpenAI | None = None
        self._sdk_available = False

        self._initialize_client()

        logger.info(
            "OPENROUTER_CLIENT_CREATED",
            sdk_available=self._sdk_available,
            model=self.model,
            api_key_present=bool(self.api_key),
            base_url=self.base_url,
        )

    def _initialize_client(self) -> None:
        """Initialize the OpenAI SDK client pointed at OpenRouter."""
        if not self.api_key:
            logger.warning("openrouter_client_unconfigured", model=self.model)
            return

        try:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            self._sdk_available = True
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("openrouter_client_init_failed", error=str(exc))
            self._client = None
            self._sdk_available = False

    def generate_text(self, prompt: str, *, max_retries: int | None = None) -> str:
        """Generate plain-text output from the provided prompt via OpenRouter."""
        if not self.api_key:
            raise OpenRouterClientError("OPENROUTER_API_KEY is not configured.")

        if not self._sdk_available or self._client is None:
            raise OpenRouterClientError("OpenRouter client is not available or failed to initialize.")

        effective_retries = max_retries or self.max_retries

        last_error: Exception | None = None
        for attempt in range(1, effective_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.choices[0].message.content
                if isinstance(text, str) and text.strip():
                    return text.strip()
                raise OpenRouterClientError("OpenRouter returned an empty response.")
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning(
                    "openrouter_generate_failed",
                    attempt=attempt,
                    retries=effective_retries,
                    error=str(exc),
                )
                last_error = exc

        if last_error is not None:
            raise OpenRouterClientError(str(last_error)) from last_error

        raise OpenRouterClientError("OpenRouter request failed without a detailed error.")
