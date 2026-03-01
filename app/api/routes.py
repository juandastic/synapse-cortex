"""
API Routes - Endpoint definitions for Synapse Cortex.
"""

import logging
import time

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from opentelemetry import trace

from app.api.dependencies import (
    ApiKeyDep,
    GenerationServiceDep,
    GraphitiDep,
    GraphServiceDep,
    HydrationServiceDep,
    IngestionServiceDep,
)
from app.core.observability import (
    anonymize_id,
    classify_error,
    mark_span_error,
    mark_span_success,
    set_span_attributes,
)
from app.schemas.models import (
    ChatCompletionRequest,
    CompilationMetadataResponse,
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
from app.services.hydration_result import CompilationMetadata
from app.services.graph_rag import (
    maybe_run_graph_rag,
    rag_outcome_to_span_attrs,
    rag_outcome_to_usage_fields,
)
from app.services.job_store import get_job, remove_job

logger = logging.getLogger(__name__)

router = APIRouter()


def _to_metadata_response(
    meta: CompilationMetadata | None,
) -> CompilationMetadataResponse | None:
    if meta is None:
        return None
    return CompilationMetadataResponse(
        is_partial=meta.is_partial,
        total_estimated_tokens=meta.total_estimated_tokens,
        included_node_ids=meta.included_node_ids,
        included_edge_ids=meta.included_edge_ids,
    )


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
    set_span_attributes(
        span,
        {
            "ingest.job_id": request.jobId,
            "ingest.user_id": request.userId,
            "ingest.session_id": request.sessionId,
            "ingest.message_count": len(request.messages),
            "ingest.total_chars": sum(len(m.content) for m in request.messages),
        },
    )
    start = time.monotonic()
    try:
        response = await ingestion_service.accept_session(request)
        set_span_attributes(
            span,
            {
                "ingest.status": response.status,
                "duration_ms": round((time.monotonic() - start) * 1000, 2),
            },
        )
        mark_span_success(span, status="skipped" if response.status == "skipped" else "success")
        return response
    except Exception as e:
        category, code = classify_error(e)
        mark_span_error(span, e, category=category, code=code)
        raise


@router.get(
    "/ingest/status/{job_id}",
    response_model=IngestStatusResponse,
    tags=["Ingest"],
)
async def ingest_status(
    job_id: str,
    _api_key: ApiKeyDep,
    hydration_service: HydrationServiceDep,
    version: Literal["v1", "v2"] = Query("v1"),
) -> IngestStatusResponse:
    """
    Poll for ingest job status. Returns full result when completed.
    
    When status is "completed", hydrates compilation from Neo4j on-demand and
    removes the job from memory. When "failed", returns error details and
    removes the job. Returns 404 if job not found.
    
    Requires X-API-SECRET header for authentication.
    """
    span = trace.get_current_span()
    start = time.monotonic()
    set_span_attributes(span, {"ingest.job_id": job_id})
    job = get_job(job_id)
    if job is None:
        mark_span_error(
            span,
            HTTPException(status_code=404, detail="Job not found"),
            category="validation",
            code="INGEST_JOB_NOT_FOUND",
            extra_attributes={"duration_ms": round((time.monotonic() - start) * 1000, 2)},
        )
        raise HTTPException(status_code=404, detail="Job not found")
    set_span_attributes(
        span,
        {
            "ingest.user_id": job.user_id,
            "ingest.session_id": job.session_id,
            "ingest.status": job.status,
        },
    )

    if job.status == "processing":
        set_span_attributes(span, {"duration_ms": round((time.monotonic() - start) * 1000, 2)})
        mark_span_success(span)
        return IngestStatusResponse(jobId=job_id, status="processing")

    if job.status == "completed":
        try:
            result = await hydration_service.build_user_knowledge(job.user_id, version=version)
        except Exception as e:
            category, code = classify_error(e)
            mark_span_error(span, e, category=category, code=code)
            raise
        ingest_metadata = None
        if (
            job.model is not None
            or job.processing_time_ms is not None
            or job.nodes_extracted is not None
            or job.edges_extracted is not None
            or job.episode_id is not None
        ):
            from app.schemas.models import IngestResponseMetadata

            ingest_metadata = IngestResponseMetadata(
                model=job.model or "unknown",
                processing_time_ms=job.processing_time_ms,
                nodes_extracted=job.nodes_extracted,
                edges_extracted=job.edges_extracted,
                episode_id=job.episode_id,
            )
            set_span_attributes(
                span,
                {
                    "ingest.metadata.model": ingest_metadata.model,
                    "ingest.metadata.processing_time_ms": ingest_metadata.processing_time_ms,
                    "ingest.metadata.nodes_extracted": ingest_metadata.nodes_extracted,
                    "ingest.metadata.edges_extracted": ingest_metadata.edges_extracted,
                    "ingest.metadata.episode_id": ingest_metadata.episode_id,
                },
            )
        remove_job(job_id)
        set_span_attributes(
            span,
            {
                "duration_ms": round((time.monotonic() - start) * 1000, 2),
                "ingest.compilation_size_chars": len(result.compilation_text),
            },
        )
        mark_span_success(span)
        return IngestStatusResponse(
            jobId=job_id,
            status="completed",
            userKnowledgeCompilation=result.compilation_text,
            compilationMetadata=_to_metadata_response(result.metadata),
            metadata=ingest_metadata,
        )

    # job.status == "failed"
    remove_job(job_id)
    set_span_attributes(
        span,
        {
            "duration_ms": round((time.monotonic() - start) * 1000, 2),
            "error.code": job.code or "INGEST_PROCESSING_ERROR",
        },
    )
    mark_span_success(span, status="failed")
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
    span = trace.get_current_span()
    start = time.monotonic()
    set_span_attributes(span, {
        "hydrate.user_id": anonymize_id(request.userId),
        "hydrate.version": request.version,
    })
    try:
        result = await hydration_service.build_user_knowledge(
            request.userId, version=request.version,
        )
        set_span_attributes(
            span,
            {
                "hydrate.success": True,
                "hydrate.compilation_size_chars": len(result.compilation_text),
                "hydrate.duration_ms": round((time.monotonic() - start) * 1000, 2),
            },
        )
        mark_span_success(span)
        return HydrateResponse(
            success=True,
            userKnowledgeCompilation=result.compilation_text,
            compilationMetadata=_to_metadata_response(result.metadata),
        )
    except Exception as e:
        category, code = classify_error(e)
        mark_span_error(
            span,
            e,
            category=category,
            code="HYDRATION_ERROR" if code == "UNEXPECTED_ERROR" else code,
            extra_attributes={
                "hydrate.success": False,
                "hydrate.duration_ms": round((time.monotonic() - start) * 1000, 2),
            },
        )
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
    graphiti: GraphitiDep,
):
    """
    OpenAI-compatible chat completions endpoint with streaming.
    
    Streams responses using Server-Sent Events (SSE) in the same format
    as OpenAI's API for easy frontend integration.
    
    When ``user_id`` and ``compilationMetadata`` are present and the graph
    was only partially loaded (``is_partial == True``), a GraphRAG
    retrieval step runs before generation to inject long-tail episodic
    memories into the payload.
    
    Requires X-API-SECRET header for authentication.
    """
    span = trace.get_current_span()
    system_prompt_chars = 0
    has_images = False
    for msg in request.messages:
        if msg.role == "system" and isinstance(msg.content, str):
            system_prompt_chars += len(msg.content)
        if isinstance(msg.content, list):
            has_images = has_images or any(part.type == "image_url" for part in msg.content)

    compilation_attrs: dict[str, object] = {}
    if request.compilationMetadata:
        cm = request.compilationMetadata
        compilation_attrs = {
            "chat.compilation.is_partial": cm.is_partial,
            "chat.compilation.estimated_tokens": cm.total_estimated_tokens,
            "chat.compilation.nodes_count": len(cm.included_node_ids),
            "chat.compilation.edges_count": len(cm.included_edge_ids),
        }

    # --- GraphRAG Context Retrieval ---
    rag_outcome = await maybe_run_graph_rag(request, graphiti)
    rag_attrs = rag_outcome_to_span_attrs(rag_outcome)
    request.rag_usage_fields = rag_outcome_to_usage_fields(rag_outcome)

    set_span_attributes(
        span,
        {
            "chat.model": request.model,
            "chat.stream": request.stream,
            "chat.messages_count": len(request.messages),
            "chat.system_prompt_chars": system_prompt_chars,
            "chat.has_images": has_images,
            **compilation_attrs,
            **rag_attrs,
        },
    )
    mark_span_success(span)
    return StreamingResponse(
        generation_service.stream_chat_completion(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
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
    span = trace.get_current_span()
    start = time.monotonic()
    set_span_attributes(span, {"graph.group_id": anonymize_id(group_id)})
    try:
        response = await graph_service.get_graph(group_id)
        set_span_attributes(
            span,
            {
                "graph.nodes_count": len(response.nodes),
                "graph.links_count": len(response.links),
                "graph.duration_ms": round((time.monotonic() - start) * 1000, 2),
            },
        )
        mark_span_success(span)
        return response
    except Exception as e:
        category, code = classify_error(e)
        mark_span_error(span, e, category=category, code=code)
        raise


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
    span = trace.get_current_span()
    start = time.monotonic()
    set_span_attributes(
        span,
        {
            "graph.correction.group_id": anonymize_id(request.group_id),
            "graph.correction.text_length_chars": len(request.correction_text),
        },
    )
    try:
        await graph_service.correct_memory(
            group_id=request.group_id,
            correction_text=request.correction_text,
        )
        set_span_attributes(
            span,
            {
                "graph.correction.success": True,
                "graph.correction.duration_ms": round((time.monotonic() - start) * 1000, 2),
            },
        )
        mark_span_success(span)
        return GraphCorrectionResponse(success=True)
    except Exception as e:
        category, code = classify_error(e)
        mark_span_error(
            span,
            e,
            category=category,
            code="MEMORY_CORRECTION_ERROR" if code == "UNEXPECTED_ERROR" else code,
            extra_attributes={
                "graph.correction.success": False,
                "graph.correction.duration_ms": round((time.monotonic() - start) * 1000, 2),
            },
        )
        logger.error(
            f"Error correcting memory for group {request.group_id}: {e}",
            exc_info=True,
        )
        return GraphCorrectionResponse(
            success=False,
            error=str(e),
            code="MEMORY_CORRECTION_ERROR",
        )
