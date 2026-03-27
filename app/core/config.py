from functools import lru_cache

from google import genai
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Neo4j Connection
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # Google Gemini API — set ONE of these. Vertex AI is preferred (uses GCP credits).
    vertex_api_key: str = ""
    google_api_key: str = ""

    # Graphiti LLM Configuration
    graphiti_model: str = "gemini-3-flash-preview"

    # API Security
    synapse_api_secret: str

    # Graphiti Configuration
    # Controls max concurrent LLM operations. Lower values help avoid 429 rate limit errors.
    # Default is 10. Reduce to 3-5 if you encounter rate limiting issues.
    semaphore_limit: int = 3

    # Axiom / OpenTelemetry (optional, tracing disabled if token is empty)
    # AXIOM_DOMAIN: base domain of your edge deployment (Settings > General in Axiom).
    # US East 1: us-east-1.aws.edge.axiom.co | EU: eu-central-1.aws.edge.axiom.co
    axiom_api_token: str = ""
    axiom_dataset: str = "synapse-cortex-traces"
    axiom_domain: str = "us-east-1.aws.edge.axiom.co"
    otel_service_name: str = "synapse-cortex"

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


def create_genai_client(settings: Settings) -> genai.Client:
    """Create a google-genai Client using Vertex AI or AI Studio based on env vars."""
    if settings.vertex_api_key:
        print("Using Vertex API Key")
        return genai.Client(
            vertexai=True,
            api_key=settings.vertex_api_key,
        )
    if settings.google_api_key:
        print("Using Google API Key")
        return genai.Client(api_key=settings.google_api_key)
    raise ValueError("Set VERTEX_API_KEY or GOOGLE_API_KEY in .env")


def create_langchain_llm(settings: Settings, **kwargs):
    """Create a LangChain chat model using Vertex AI or AI Studio based on env vars."""
    if settings.vertex_api_key:
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            api_key=settings.vertex_api_key,
            **kwargs,
        )
    if settings.google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            **kwargs,
        )
    raise ValueError("Set VERTEX_API_KEY or GOOGLE_API_KEY in .env")
