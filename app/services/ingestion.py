"""
Ingestion Service - Processes chat sessions via Graphiti and returns user knowledge compilation.

Combines:
1. Graphiti add_episode for knowledge extraction
2. Hydration service for building the updated compilation (on-demand via GET /ingest/status)

Async flow: accept_session returns 202 immediately, _process_background runs in background.
"""

import asyncio
import logging
import time
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from opentelemetry import trace

from app.schemas.models import (
    IngestAcceptedResponse,
    IngestMessage,
    IngestRequest,
    IngestResponse,
    IngestResponseMetadata,
)
from app.services.hydration import HydrationService
from app.services.job_store import complete_job, create_job, fail_job, get_job

logger = logging.getLogger(__name__)

# Minimum requirements for ingestion
MIN_MESSAGES = 1
MIN_TOTAL_CHARS = 5


class IngestionService:
    """Service for processing chat sessions into the knowledge graph."""

    def __init__(self, graphiti: Graphiti, hydration_service: HydrationService, model: str):
        self.graphiti = graphiti
        self.hydration_service = hydration_service
        self.model = model

    async def accept_session(self, request: IngestRequest) -> IngestAcceptedResponse:
        """
        Accept an ingest request: validate, create job entry, launch background processing.
        Returns immediately with 202. If messages are insufficient, returns "skipped" with compilation.
        """
        # Validation: skip if messages too short
        if not self._should_ingest(request.messages):
            logger.info(
                f"Skipping ingestion for session {request.sessionId}: "
                f"insufficient messages ({len(request.messages)} messages)"
            )
            # Still hydrate to return existing knowledge (fast path)
            compilation = await self.hydration_service.build_user_knowledge(
                request.userId
            )
            return IngestAcceptedResponse(
                jobId=request.jobId,
                status="skipped",
                userKnowledgeCompilation=compilation,
            )

        # Idempotency: if job already exists (duplicate submit), return current status
        if not create_job(request.jobId, request.userId, request.sessionId):
            job = get_job(request.jobId)
            assert job is not None  # create_job returned False so it exists
            return IngestAcceptedResponse(
                jobId=request.jobId,
                status="processing",
            )

        # Launch background processing
        asyncio.create_task(self._process_background(request.jobId, request))

        return IngestAcceptedResponse(
            jobId=request.jobId,
            status="processing",
        )

    async def _process_background(self, job_id: str, request: IngestRequest) -> None:
        """
        Background task: run Graphiti add_episode, update job store on completion/failure.
        Does NOT run hydration (that happens on GET when status is "completed").
        """
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "ingest.process_background",
            attributes={
                "ingest.job_id": job_id,
                "ingest.user_id": request.userId,
                "ingest.session_id": request.sessionId,
            },
        ):
            try:
                episode_content = self._format_messages_for_graphiti(request.messages)
                episode_name = f"session_{request.sessionId}"

                logger.info(f"Adding episode {episode_name} for user {request.userId}")

                start_time = time.monotonic()
                result = await self.graphiti.add_episode(
                    name=episode_name,
                    episode_body=episode_content,
                    source=EpisodeType.message,
                    source_description="Chat conversation from Synapse AI Chat application",
                    group_id=request.userId,
                    reference_time=datetime.fromtimestamp(
                        request.metadata.sessionEndedAt / 1000
                    ),
                )
                elapsed_ms = (time.monotonic() - start_time) * 1000

                # Update job store with completion metadata (no hydration here)
                complete_job(
                    job_id,
                    model=self.model,
                    processing_time_ms=round(elapsed_ms, 1),
                    nodes_extracted=len(result.nodes),
                    edges_extracted=len(result.edges),
                    episode_id=result.episode.uuid,
                )

                logger.info(
                    f"Successfully processed session {request.sessionId} "
                    f"for user {request.userId} "
                    f"({len(result.nodes)} nodes, {len(result.edges)} edges, "
                    f"{elapsed_ms:.0f}ms)"
                )

            except Exception as e:
                logger.error(
                    f"Error processing session {request.sessionId}: {e}",
                    exc_info=True,
                )
                fail_job(job_id, error=str(e), code="GRAPH_PROCESSING_ERROR")

    def _should_ingest(self, messages: list[IngestMessage]) -> bool:
        """Check if the session meets minimum requirements for ingestion."""
        if len(messages) < MIN_MESSAGES:
            return False

        total_chars = sum(len(msg.content) for msg in messages)
        if total_chars < MIN_TOTAL_CHARS:
            return False

        return True

    def _format_messages_for_graphiti(self, messages: list[IngestMessage]) -> str:
        """Format messages into a single episode body for Graphiti."""
        formatted_lines = []
        for msg in messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            formatted_lines.append(f"{role_label}: {msg.content}")

        return "\n\n".join(formatted_lines)
