"""
Ingestion Service - Processes chat sessions via Graphiti and returns user knowledge compilation.

Combines:
1. Graphiti add_episode for knowledge extraction
2. Hydration service for building the updated compilation
"""

import logging
import time
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from app.schemas.models import IngestMessage, IngestRequest, IngestResponse, IngestResponseMetadata
from app.services.hydration import HydrationService

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

    async def process_session(self, request: IngestRequest) -> IngestResponse:
        """
        Process a chat session: ingest via Graphiti, then hydrate.

        Args:
            request: The ingest request containing messages and metadata.

        Returns:
            IngestResponse with success status and user knowledge compilation.
        """
        try:
            # Validation: skip if messages too short
            if not self._should_ingest(request.messages):
                logger.info(
                    f"Skipping ingestion for session {request.sessionId}: "
                    f"insufficient messages ({len(request.messages)} messages)"
                )
                # Still hydrate to return existing knowledge
                compilation = await self.hydration_service.build_user_knowledge(
                    request.userId
                )
                return IngestResponse(
                    success=True,
                    userKnowledgeCompilation=compilation,
                    metadata=IngestResponseMetadata(model=self.model),
                )

            # Process via Graphiti
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

            # Hydrate: Build updated userKnowledgeCompilation
            compilation = await self.hydration_service.build_user_knowledge(
                request.userId
            )

            logger.info(
                f"Successfully processed session {request.sessionId} "
                f"for user {request.userId} "
                f"({len(result.nodes)} nodes, {len(result.edges)} edges, "
                f"{elapsed_ms:.0f}ms)"
            )

            return IngestResponse(
                success=True,
                userKnowledgeCompilation=compilation,
                metadata=IngestResponseMetadata(
                    model=self.model,
                    processing_time_ms=round(elapsed_ms, 1),
                    nodes_extracted=len(result.nodes),
                    edges_extracted=len(result.edges),
                    episode_id=result.episode.uuid,
                ),
            )

        except Exception as e:
            logger.error(
                f"Error processing session {request.sessionId}: {e}",
                exc_info=True,
            )
            return IngestResponse(
                success=False,
                error=str(e),
                code="GRAPH_PROCESSING_ERROR",
                metadata=IngestResponseMetadata(model=self.model),
            )

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
