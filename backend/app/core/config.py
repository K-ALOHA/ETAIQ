"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings for the ETAIQ backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="ETAIQ API", description="Application display name")
    app_version: str = Field(default="0.1.0", description="Semantic application version")
    environment: str = Field(default="development", description="Runtime environment name")
    debug: bool = Field(default=False, description="Enable debug mode")
    api_prefix: str = Field(default="/api/v1", description="Base API route prefix")

    database_url: str = Field(
        default="postgresql://etaiq:etaiq@localhost:5432/etaiq",
        description="PostgreSQL connection string",
    )
    jwt_secret: str = Field(default="change-me-in-production", description="JWT signing secret")
    openai_api_key: str = Field(default="", description="OpenAI API key for AI assistant")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    model_path: str = Field(default="./ml/artifacts", description="Path to ML model artifacts")
    log_level: str = Field(default="INFO", description="Logging verbosity level")

    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins for the frontend",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Returns:
        Settings: Singleton application settings loaded from the environment.
    """
    return Settings()
