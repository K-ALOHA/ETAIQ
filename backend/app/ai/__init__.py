"""AI assistant package for ETAIQ."""

from app.ai.assistant import ChatService, ETAIQAssistantService
from app.ai.context import AIContextBuilder, ContextBuilder
from app.ai.conversation import ConversationManager
from app.ai.openrouter_client import OpenRouterClient, OpenRouterClientError
from app.ai.prompts import PromptBuilder
from app.ai.schemas import AssistantRequest, AssistantResponse

__all__ = [
    "AIContextBuilder",
    "AssistantRequest",
    "AssistantResponse",
    "ChatService",
    "ContextBuilder",
    "ConversationManager",
    "ETAIQAssistantService",
    "OpenRouterClient",
    "OpenRouterClientError",
    "PromptBuilder",
]
