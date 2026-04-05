"""
Synapse Cortex - Main Application Entrypoint

A stateless REST API for knowledge graph operations and LLM interactions.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from neo4j import AsyncGraphDatabase

from app.api.routes import router
from app.core.config import create_genai_client, create_posthog_genai_client, get_settings
from app.core.posthog import get_posthog, init_posthog, set_posthog_genai_client, shutdown_posthog
from app.core.telemetry import setup_telemetry, shutdown_telemetry
from app.services.generation import GenerationService
from app.services.graph import GraphService
from app.services.hydration import HydrationService
from app.services.ingestion import IngestionService
from app.services.notion_correction import NotionCorrectionService
from app.services.notion_export import NotionExportService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Initializes database connections and services on startup,
    cleans up on shutdown.
    """
    settings = get_settings()
    logger.info("Starting Synapse Cortex...")

    # Initialize PostHog LLM Analytics (optional)
    init_posthog(settings)

    # Initialize Neo4j async driver
    neo4j_driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    # Verify Neo4j connection
    try:
        await neo4j_driver.verify_connectivity()
        logger.info("Connected to Neo4j")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        raise

    # Create GenAI clients.
    # If PostHog is configured, use its wrapper for auto-tracking LLM calls;
    # extract the underlying raw client for Graphiti (which needs genai.Client).
    # If PostHog is not configured, fall back to a plain genai.Client.
    posthog_client = get_posthog()
    if posthog_client:
        posthog_genai_client, raw_genai_client = create_posthog_genai_client(
            settings, posthog_client,
        )
        # Use the PostHog-wrapped client everywhere — it has .aio.models shim
        # so Graphiti's internal calls (client.aio.models.generate_content)
        # are automatically tracked in PostHog.
        graphiti_client = posthog_genai_client
        generation_client = posthog_genai_client
        set_posthog_genai_client(posthog_genai_client)
        logger.info("Using PostHog-wrapped GenAI client for LLM analytics")
    else:
        raw_genai_client = create_genai_client(settings)
        graphiti_client = raw_genai_client
        generation_client = raw_genai_client

    # Initialize Graphiti — uses PostHog-wrapped client when available,
    # so all internal LLM calls (entity extraction, reranking, etc.) are tracked.
    graphiti = Graphiti(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        llm_client=GeminiClient(
            config=LLMConfig(model=settings.graphiti_model),
            client=graphiti_client,
        ),
        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(embedding_model="gemini-embedding-001"),
            client=raw_genai_client,
        ),
        cross_encoder=GeminiRerankerClient(
            config=LLMConfig(model=settings.graphiti_model),
            client=graphiti_client,
        ),
        max_coroutines=settings.semaphore_limit,
    )
    logger.info(f"Graphiti initialized with model={settings.graphiti_model}, max_coroutines={settings.semaphore_limit}")

    # Build indices and constraints (safe to call multiple times, only creates if missing)
    await graphiti.build_indices_and_constraints()
    logger.info("Graphiti indices and constraints initialized")

    # Initialize services
    hydration_service = HydrationService(neo4j_driver)
    ingestion_service = IngestionService(graphiti, settings.graphiti_model)
    generation_service = GenerationService(generation_client)
    graph_service = GraphService(neo4j_driver, graphiti)
    notion_export_service = NotionExportService(
        hydration_service=hydration_service,
        settings=settings,
    )
    notion_correction_service = NotionCorrectionService(
        graphiti=graphiti,
        settings=settings,
    )

    # Store in app state for dependency injection
    app.state.neo4j_driver = neo4j_driver
    app.state.graphiti = graphiti
    app.state.hydration_service = hydration_service
    app.state.ingestion_service = ingestion_service
    app.state.generation_service = generation_service
    app.state.graph_service = graph_service
    app.state.notion_export_service = notion_export_service
    app.state.notion_correction_service = notion_correction_service

    logger.info("Synapse Cortex started successfully")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Synapse Cortex...")
    await neo4j_driver.close()
    await graphiti.close()
    shutdown_posthog()
    shutdown_telemetry()
    logger.info("Synapse Cortex shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Synapse Cortex",
    description="Cognitive backend for the Synapse AI Chat application",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# OpenTelemetry instrumentation (after app is fully configured)
setup_telemetry(app)
