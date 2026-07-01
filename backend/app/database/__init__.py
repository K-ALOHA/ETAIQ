"""Database connection and session management (placeholder)."""

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


async def init_db() -> None:
    """Initialize database connections when persistence is enabled.

    This function is a placeholder for future PostgreSQL integration.
    """
    logger.info("database_init_skipped", reason="not_configured_in_milestone_1")
