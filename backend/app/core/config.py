"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings for the ETAIQ backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Disable complex env parsing (prevents cors_origins from being JSON-parsed)
        env_file_case_insensitive=True,
    )

    app_name: str = Field(default="ETAIQ API", description="Application display name")
    app_version: str = Field(default="0.1.0", description="Semantic application version")
    environment: str = Field(default="development", description="Runtime environment name")
    debug: bool = Field(default=False, description="Enable debug mode")
    api_prefix: str = Field(default="/api/v1", description="Base API route prefix")

    database_url: str = Field(
        default="",
        description="PostgreSQL connection string",
    )
    jwt_secret: str = Field(
        default="",
        description="JWT signing secret (REQUIRED - must be set in environment)",
    )
    openrouter_api_key: str = Field(default="", description="OpenRouter API key for AI assistant")
    openrouter_model: str = Field(
        default="deepseek/deepseek-chat-v3", description="OpenRouter model name"
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", description="OpenRouter API base URL"
    )
    model_path: str = Field(default="./ml/artifacts", description="Path to ML model artifacts")
    model_artifact_dir: str = Field(
        default="./ml/artifacts/models",
        description="Directory containing .joblib model files",
    )
    log_level: str = Field(default="INFO", description="Logging verbosity level")

    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:3001"],
        description=(
            "Allowed CORS origins for the frontend"
            " (use comma-separated list in CORS_ORIGINS env var)"
        ),
        json_schema_extra={"env_parse": False}
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins: split by commas if string, return as-is if list."""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # Split by commas and strip
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        """Ensure JWT_SECRET is set in production-like environments."""
        if not v or v.isspace():
            raise ValueError(
                "JWT_SECRET environment variable must be set. "
                "This is a security-critical value that must be configured per environment."
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Returns:
        Settings: Singleton application settings loaded from the environment.
    """
    return Settings()
