from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Neo4j Connection
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # Google Gemini API
    google_api_key: str

    # API Security
    synapse_api_secret: str

    # Graphiti Configuration
    # Controls max concurrent LLM operations. Lower values help avoid 429 rate limit errors.
    # Default is 10. Reduce to 3-5 if you encounter rate limiting issues.
    semaphore_limit: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
