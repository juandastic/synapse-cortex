"""Construct the API's Graphiti client with explicit provider selection."""

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient

from app.core.config import Settings
from app.core.graphiti_llm import OpenRouterGraphitiClient


def create_graphiti(settings: Settings, raw_genai_client, tracked_genai_client=None) -> Graphiti:
    genai_client = tracked_genai_client or raw_genai_client
    config = LLMConfig(
        model=settings.graphiti_model,
        small_model=settings.graphiti_small_model,
        max_tokens=settings.graphiti_max_tokens,
    )
    if settings.graphiti_provider == "openrouter":
        if not settings.open_router_api_key:
            raise ValueError("OPEN_ROUTER_API_KEY is required when GRAPHITI_PROVIDER=openrouter")
        if not all(model.startswith("openai/") for model in (config.model, config.small_model)):
            raise ValueError("OpenRouter graph models must use OpenAI model IDs (openai/...)")
        config.api_key = settings.open_router_api_key
        llm_client = OpenRouterGraphitiClient(
            config,
            reasoning_effort=settings.graphiti_reasoning_effort,
            small_reasoning_effort=settings.graphiti_small_reasoning_effort,
        )
        # RRF searches do not use the reranker; retain Gemini's existing adapter.
        cross_encoder = GeminiRerankerClient(client=genai_client)
    else:
        llm_client = GeminiClient(config=config, client=genai_client)
        cross_encoder = GeminiRerankerClient(
            config=LLMConfig(model=settings.graphiti_model), client=genai_client
        )
    return Graphiti(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        llm_client=llm_client,
        # Keep the existing vector space. Changing this requires re-embedding the graph.
        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(embedding_model="gemini-embedding-001"),
            client=raw_genai_client,
        ),
        cross_encoder=cross_encoder,
        max_coroutines=settings.semaphore_limit,
    )
