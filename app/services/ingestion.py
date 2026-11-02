"""
Ingestion Service - Processes chat sessions via Graphiti.

Graphiti add_episode extracts entities and relationships into Neo4j.
Published hydration builds after evidence finalization; status reads it or uses the v2 fallback.

Async flow: accept_session returns 202 immediately, _process_background runs in background.
"""

import asyncio
import logging
import time
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EpisodeType
from graphiti_core.prompts.models import Message
from opentelemetry import trace
from pydantic import BaseModel

from app.core.observability import (
    classify_error,
    mark_span_error,
    mark_span_success,
    set_span_attributes,
)
from app.core.posthog import capture_span, capture_trace, new_trace_id, posthog_user_context
from app.schemas.models import (
    IngestAcceptedResponse,
    IngestMessage,
    IngestRequest,
)
from app.services.job_store import complete_job, create_job, fail_job, get_job

logger = logging.getLogger(__name__)

# Minimum requirements for ingestion
MIN_MESSAGES = 1
MIN_TOTAL_CHARS = 5

# Assistant messages beyond this limit are truncated before sending to Graphiti.
# User messages carry the signal (personal details, decisions, experiences);
# assistant responses are verbose and add noise to entity extraction.
ASSISTANT_TRUNCATE_LIMIT = 300

SUMMARY_PROMPT = (
    "Summarize this conversation in 2-3 sentences in the same language as the conversation. "
    "Focus on: key topics discussed, decisions made, and any important insights. "
    "Be factual and concise. Return the summary in the requested JSON schema."
)

# Cap input to the summarization call to control cost
SUMMARY_INPUT_LIMIT = 8000
SUMMARY_TIMEOUT_SECONDS = 30
SUMMARY_STORE_TIMEOUT_SECONDS = 10


class EpisodeSummary(BaseModel):
    summary: str


class IngestionService:
    """Service for processing chat sessions into the knowledge graph."""

    def __init__(self, graphiti: Graphiti, model: str, hydration_changed=None):
        self.graphiti = graphiti
        self.model = model
        self.hydration_changed = hydration_changed

    async def accept_session(self, request: IngestRequest) -> IngestAcceptedResponse:
        """
        Accept an ingest request: validate, create job entry, launch background processing.
        Returns immediately with 202. If messages are insufficient, returns "skipped" —
        the frontend uses its cached knowledge as fallback.
        """
        if not self._should_ingest(request.messages):
            logger.info(
                f"Skipping ingestion for session {request.sessionId}: "
                f"insufficient messages ({len(request.messages)} messages)"
            )
            return IngestAcceptedResponse(
                jobId=request.jobId,
                status="skipped",
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
        Queues hydration only after finalizing evidence; never waits for compilation.
        """
        tracer = trace.get_tracer(__name__)
        posthog_trace_id = new_trace_id()
        with tracer.start_as_current_span(
            "ingest.process_background",
            attributes={
                "ingest.job_id": job_id,
                "ingest.user_id": request.userId,
                "ingest.session_id": request.sessionId,
            },
        ) as span:
            start = time.monotonic()
            episode_content = self._format_messages_for_graphiti(request.messages)
            try:
                episode_name = f"session_{request.sessionId}"

                logger.info(f"Adding episode {episode_name} for user {request.userId}")

                start_time = time.monotonic()
                with posthog_user_context(request.userId, posthog_trace_id, request.sessionId):
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

                logger.info(
                    f"Successfully processed session {request.sessionId} "
                    f"for user {request.userId} "
                    f"({len(result.nodes)} nodes, {len(result.edges)} edges, "
                    f"{elapsed_ms:.0f}ms)"
                )
                set_span_attributes(
                    span,
                    {
                        "ingest.status": "completed",
                        "ingest.processing_time_ms": round(elapsed_ms, 1),
                        "ingest.nodes_extracted": len(result.nodes),
                        "ingest.edges_extracted": len(result.edges),
                        "ingest.episode_id": result.episode.uuid,
                        "duration_ms": round((time.monotonic() - start) * 1000, 2),
                    },
                )
                mark_span_success(span)

                # PostHog LLM Analytics
                capture_trace(
                    request.userId,
                    posthog_trace_id,
                    name="ingestion",
                    session_id=request.sessionId,
                    properties={"pipeline": "ingestion"},
                )
                capture_span(
                    request.userId,
                    posthog_trace_id,
                    name="graphiti.add_episode",
                    input_data=episode_content[:2000],
                    output_data=f"{len(result.nodes)} nodes, {len(result.edges)} edges",
                    duration_ms=elapsed_ms,
                )

                # Summarize the episode and store on the Neo4j node.
                # Finalize evidence before publishing completion or queuing compilation.
                if result.episode:
                    try:
                        with posthog_user_context(
                            request.userId, posthog_trace_id, request.sessionId
                        ):
                            summary = await asyncio.wait_for(
                                self._summarize_episode(episode_content),
                                timeout=SUMMARY_TIMEOUT_SECONDS,
                            )
                        if summary:
                            await asyncio.wait_for(
                                self._store_episode_summary(result.episode.uuid, summary),
                                timeout=SUMMARY_STORE_TIMEOUT_SECONDS,
                            )
                            logger.info(f"Stored episode summary ({len(summary)} chars)")
                    except Exception:
                        logger.warning(
                            "Optional episode summary failed; graph ingestion committed",
                            exc_info=True,
                        )

                if self.hydration_changed is not None:
                    try:
                        await self.hydration_changed(request.userId)
                    except Exception:
                        # Graph ingestion succeeded. /hydrate also bootstraps missing work.
                        logger.exception("Could not enqueue hydration; next read will retry")
                complete_job(
                    job_id,
                    model=self.model,
                    processing_time_ms=round((time.monotonic() - start) * 1000, 1),
                    nodes_extracted=len(result.nodes),
                    edges_extracted=len(result.edges),
                    episode_id=result.episode.uuid,
                )

            except Exception as e:
                category, code = classify_error(e)
                mark_span_error(
                    span,
                    e,
                    category=category,
                    code="GRAPH_PROCESSING_ERROR" if code == "UNEXPECTED_ERROR" else code,
                    extra_attributes={
                        "ingest.status": "failed",
                        "duration_ms": round((time.monotonic() - start) * 1000, 2),
                    },
                )
                logger.error(
                    f"Error processing session {request.sessionId}: {e}",
                    exc_info=True,
                )
                fail_job(job_id, error=str(e), code="GRAPH_PROCESSING_ERROR")

                # PostHog LLM Analytics (error)
                capture_trace(
                    request.userId,
                    posthog_trace_id,
                    name="ingestion",
                    session_id=request.sessionId,
                    properties={"pipeline": "ingestion"},
                )
                capture_span(
                    request.userId,
                    posthog_trace_id,
                    name="graphiti.add_episode",
                    input_data=episode_content[:2000],
                    duration_ms=(time.monotonic() - start) * 1000,
                    properties={"$ai_is_error": True, "$ai_error": str(e)[:500]},
                )

    async def _summarize_episode(self, episode_content: str) -> str | None:
        """Generate a concise summary from the already-formatted episode text.

        Reuses the output of _format_messages_for_graphiti (which already has
        truncated assistant messages), capped at SUMMARY_INPUT_LIMIT chars.
        Returns None on failure.
        """
        text = episode_content[:SUMMARY_INPUT_LIMIT]
        try:
            response = await self.graphiti.llm_client.generate_response(
                [
                    Message(role="system", content=SUMMARY_PROMPT),
                    Message(role="user", content=text),
                ],
                response_model=EpisodeSummary,
                model_size=ModelSize.small,
                max_tokens=1024,
                prompt_name="ingestion.episode_summary",
            )
            return response["summary"].strip() or None
        except Exception as e:
            logger.warning(f"Episode summarization failed: {e}")
            return None

    async def _store_episode_summary(self, episode_uuid: str, summary: str) -> None:
        """Write the summary as a custom property on the Episodic node in Neo4j."""
        try:
            async with self.graphiti.driver.session() as session:
                await session.run(
                    "MATCH (e:Episodic {uuid: $uuid}) SET e.summary = $summary",
                    uuid=episode_uuid,
                    summary=summary,
                )
        except Exception as e:
            logger.warning(f"Failed to store episode summary: {e}")

    def _should_ingest(self, messages: list[IngestMessage]) -> bool:
        """Check if the session meets minimum requirements for ingestion."""
        if len(messages) < MIN_MESSAGES:
            return False

        total_chars = sum(len(msg.content) for msg in messages)
        if total_chars < MIN_TOTAL_CHARS:
            return False

        return True

    def _format_messages_for_graphiti(self, messages: list[IngestMessage]) -> str:
        """Format messages into a single episode body for Graphiti.

        Assistant messages are truncated to reduce noise — user messages
        carry the valuable memory signal (personal details, decisions, experiences).
        """
        formatted_lines = []
        for msg in messages:
            if msg.role == "user":
                formatted_lines.append(f"User: {msg.content}")
            else:
                content = msg.content
                if len(content) > ASSISTANT_TRUNCATE_LIMIT:
                    content = content[:ASSISTANT_TRUNCATE_LIMIT] + "..."
                formatted_lines.append(f"Assistant: {content}")

        return "\n\n".join(formatted_lines)
