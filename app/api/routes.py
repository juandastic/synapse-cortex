"""
API Routes - Endpoint definitions for Synapse Cortex.
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from opentelemetry import trace

from app.api.dependencies import (
    ApiKeyDep,
    GenerationServiceDep,
    GraphServiceDep,
    HydrationServiceDep,
    IngestionServiceDep,
)
from app.schemas.models import (
    ChatCompletionRequest,
    GraphCorrectionRequest,
    GraphCorrectionResponse,
    GraphResponse,
    HealthResponse,
    HydrateRequest,
    HydrateResponse,
    IngestAcceptedResponse,
    IngestRequest,
    IngestStatusResponse,
)
from app.services.job_store import get_job, remove_job

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint for load balancers and monitoring.
    
    No authentication required.
    """
    return HealthResponse()


@router.post(
    "/ingest",
    response_model=IngestAcceptedResponse,
    status_code=202,
    tags=["Ingest"],
)
async def ingest_session(
    request: IngestRequest,
    _api_key: ApiKeyDep,
    ingestion_service: IngestionServiceDep,
) -> IngestAcceptedResponse:
    """
    Accept a chat session for async processing (fire-and-forget).
    
    Returns 202 immediately with jobId and status. Poll GET /ingest/status/{jobId}
    to check completion. If messages are insufficient, returns "skipped" with
    compilation immediately.
    
    Requires X-API-SECRET header for authentication.
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("ingest.job_id", request.jobId)
        span.set_attribute("ingest.user_id", request.userId)
        span.set_attribute("ingest.session_id", request.sessionId)
    return await ingestion_service.accept_session(request)


@router.get(
    "/ingest/status/{job_id}",
    response_model=IngestStatusResponse,
    tags=["Ingest"],
)
async def ingest_status(
    job_id: str,
    _api_key: ApiKeyDep,
    hydration_service: HydrationServiceDep,
) -> IngestStatusResponse:
    """
    Poll for ingest job status. Returns full result when completed.
    
    When status is "completed", hydrates compilation from Neo4j on-demand and
    removes the job from memory. When "failed", returns error details and
    removes the job. Returns 404 if job not found.
    
    Requires X-API-SECRET header for authentication.
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("ingest.job_id", job_id)
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if span.is_recording():
        span.set_attribute("ingest.user_id", job.user_id)
        span.set_attribute("ingest.session_id", job.session_id)
        span.set_attribute("ingest.status", job.status)

    if job.status == "processing":
        return IngestStatusResponse(jobId=job_id, status="processing")

    if job.status == "completed":
        # Hydrate on-demand from Neo4j
        compilation = await hydration_service.build_user_knowledge(job.user_id)
        metadata = None
        if (
            job.model is not None
            or job.processing_time_ms is not None
            or job.nodes_extracted is not None
            or job.edges_extracted is not None
            or job.episode_id is not None
        ):
            from app.schemas.models import IngestResponseMetadata

            metadata = IngestResponseMetadata(
                model=job.model or "unknown",
                processing_time_ms=job.processing_time_ms,
                nodes_extracted=job.nodes_extracted,
                edges_extracted=job.edges_extracted,
                episode_id=job.episode_id,
            )
        remove_job(job_id)
        return IngestStatusResponse(
            jobId=job_id,
            status="completed",
            userKnowledgeCompilation=compilation,
            metadata=metadata,
        )

    # job.status == "failed"
    remove_job(job_id)
    return IngestStatusResponse(
        jobId=job_id,
        status="failed",
        error=job.error,
        code=job.code,
    )


@router.post("/hydrate", response_model=HydrateResponse, tags=["Debug"])
async def hydrate_user(
    request: HydrateRequest,
    _api_key: ApiKeyDep,
    hydration_service: HydrationServiceDep,
) -> HydrateResponse:
    """
    Get the current user knowledge compilation without processing new data.
    
    This is a read-only endpoint useful for:
    - Debugging the current state of a user's knowledge graph
    - Fetching the compilation without re-indexing
    - Testing the hydration logic
    
    Requires X-API-SECRET header for authentication.
    """
    try:
        compilation = await hydration_service.build_user_knowledge(request.userId)
        return HydrateResponse(
            success=True,
            userKnowledgeCompilation=compilation,
        )
    except Exception as e:
        logger.error(f"Error hydrating user {request.userId}: {e}", exc_info=True)
        return HydrateResponse(
            success=False,
            error=str(e),
            code="HYDRATION_ERROR",
        )


@router.post("/v1/chat/completions", tags=["Chat"])
async def chat_completions(
    request: ChatCompletionRequest,
    _api_key: ApiKeyDep,
    generation_service: GenerationServiceDep,
):
    """
    OpenAI-compatible chat completions endpoint with streaming.
    
    Streams responses using Server-Sent Events (SSE) in the same format
    as OpenAI's API for easy frontend integration.
    
    Requires X-API-SECRET header for authentication.
    """
    return StreamingResponse(
        generation_service.stream_chat_completion(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/v1/graph/{group_id}", response_model=GraphResponse, tags=["Graph"])
async def get_graph(
    group_id: str,
    _api_key: ApiKeyDep,
    graph_service: GraphServiceDep,
) -> GraphResponse:
    """
    Retrieve the knowledge graph for a user in react-force-graph format.

    Returns nodes (entities) and links (relationships) suitable for
    rendering with react-force-graph-2d. Each node includes a `val` field
    representing its connection count for automatic visual sizing.

    Only returns currently valid relationships (non-expired) and excludes
    Episodic nodes.

    Requires X-API-SECRET header for authentication.
    """
    return await graph_service.get_graph(group_id)


@router.post(
    "/v1/graph/correction",
    response_model=GraphCorrectionResponse,
    tags=["Graph"],
)
async def correct_memory(
    request: GraphCorrectionRequest,
    _api_key: ApiKeyDep,
    graph_service: GraphServiceDep,
) -> GraphCorrectionResponse:
    """
    Apply a natural language memory correction via Graphiti.

    Instead of direct CRUD operations on graph nodes (which would break
    embeddings and temporal integrity), this endpoint feeds the correction
    text through Graphiti's add_episode pipeline. Graphiti will automatically
    invalidate outdated edges and create new ones.

    Example correction: "Ya no quiero aplicar a la Visa O-1, decidí quedarme
    en Colombia por ahora."

    Requires X-API-SECRET header for authentication.
    """
    try:
        await graph_service.correct_memory(
            group_id=request.group_id,
            correction_text=request.correction_text,
        )
        return GraphCorrectionResponse(success=True)
    except Exception as e:
        logger.error(
            f"Error correcting memory for group {request.group_id}: {e}",
            exc_info=True,
        )
        return GraphCorrectionResponse(
            success=False,
            error=str(e),
            code="MEMORY_CORRECTION_ERROR",
        )
