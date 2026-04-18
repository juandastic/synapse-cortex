import json
import logging
from functools import lru_cache

from google import genai
from google.oauth2 import service_account
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Neo4j Connection
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # Google Gemini API auth. Full Vertex AI (GCP_PROJECT + credentials) is
    # required for context caching. VERTEX_API_KEY uses Express Mode which
    # does not support the caching API. GOOGLE_API_KEY uses AI Studio.
    gcp_project: str = ""
    gcp_location: str = "global"
    gcp_credentials_json: str = ""
    vertex_api_key: str = ""
    google_api_key: str = ""

    # Graphiti LLM Configuration (entity extraction, reranking — small prompts)
    graphiti_model: str = "gemini-3-flash-preview"
    # Chat model for /v1/chat/completions. MUST match what the client sends,
    # because Gemini caches are model-bound: a cache created for model A
    # cannot be used with model B (400 INVALID_ARGUMENT).
    chat_model: str = "gemini-3.1-pro-preview"

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

    # PostHog LLM Analytics (optional, disabled if api key is empty)
    posthog_api_key: str = ""
    posthog_host: str = "https://us.i.posthog.com"

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


def _load_vertex_credentials(settings: Settings):
    """Parse GCP_CREDENTIALS_JSON into a Credentials object, or None to fall
    back to Application Default Credentials."""
    if not settings.gcp_credentials_json:
        return None
    try:
        info = json.loads(settings.gcp_credentials_json)
    except json.JSONDecodeError as e:
        raise ValueError(
            "GCP_CREDENTIALS_JSON is set but not valid JSON."
        ) from e
    return service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def create_genai_client(settings: Settings) -> genai.Client:
    """Create a google-genai Client using Vertex AI or AI Studio based on env vars."""
    if settings.gcp_project:
        credentials = _load_vertex_credentials(settings)
        auth_source = "GCP_CREDENTIALS_JSON" if credentials else "ADC (file or gcloud login)"
        logger.info(
            "GenAI auth: Vertex AI (full) | project=%s location=%s auth=%s | caching=supported",
            settings.gcp_project, settings.gcp_location, auth_source,
        )
        kwargs: dict = {
            "vertexai": True,
            "project": settings.gcp_project,
            "location": settings.gcp_location,
        }
        if credentials is not None:
            kwargs["credentials"] = credentials
        return genai.Client(**kwargs)
    if settings.vertex_api_key:
        logger.info(
            "GenAI auth: Vertex AI Express (VERTEX_API_KEY) | "
            "caching=NOT_SUPPORTED (use GCP_PROJECT for caching)"
        )
        return genai.Client(vertexai=True, api_key=settings.vertex_api_key)
    if settings.google_api_key:
        logger.info(
            "GenAI auth: Google AI Studio (GOOGLE_API_KEY) | caching=supported"
        )
        return genai.Client(api_key=settings.google_api_key)
    raise ValueError("Set GCP_PROJECT, VERTEX_API_KEY, or GOOGLE_API_KEY in .env")


class _AioShim:
    """Shim that exposes `client.aio.models` pointing to PostHog's AsyncModels.

    Graphiti calls `client.aio.models.generate_content(...)`.
    PostHog's AsyncClient exposes that as `client.models.generate_content(...)`.
    This bridges the gap so both interfaces work on the same tracked object.
    """

    def __init__(self, models):
        self.models = models


def create_posthog_genai_client(settings: Settings, posthog_client):
    """Create a PostHog-wrapped async GenAI client for automatic LLM tracking.

    Returns (wrapped_client, raw_genai_client).
    The wrapped_client has both `client.models` and `client.aio.models` pointing
    to PostHog's tracked AsyncModels, so Graphiti's internal calls get auto-tracked.
    """
    from posthog.ai.gemini import AsyncClient

    kwargs = {}
    if settings.gcp_project:
        kwargs.update(
            vertexai=True,
            project=settings.gcp_project,
            location=settings.gcp_location,
        )
        credentials = _load_vertex_credentials(settings)
        if credentials is not None:
            kwargs["credentials"] = credentials
    elif settings.vertex_api_key:
        kwargs.update(vertexai=True, api_key=settings.vertex_api_key)
    elif settings.google_api_key:
        kwargs["api_key"] = settings.google_api_key
    else:
        raise ValueError("Set GCP_PROJECT, VERTEX_API_KEY, or GOOGLE_API_KEY in .env")

    wrapped = AsyncClient(
        posthog_client=posthog_client,
        **kwargs,
    )
    # Add .aio.models shim so Graphiti (which calls client.aio.models.generate_content)
    # routes through PostHog's tracked AsyncModels
    wrapped.aio = _AioShim(wrapped.models)
    # The underlying genai.Client for anything that truly needs the raw client
    raw_client = wrapped.models._client
    return wrapped, raw_client


def create_langchain_llm(settings: Settings, **kwargs):
    """Create a LangChain chat model using Vertex AI or AI Studio based on env vars."""
    if settings.gcp_project:
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            project=settings.gcp_project,
            location=settings.gcp_location,
            **kwargs,
        )
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
    raise ValueError("Set GCP_PROJECT, VERTEX_API_KEY, or GOOGLE_API_KEY in .env")
