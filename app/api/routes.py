"""
API Routes - Endpoint definitions for Synapse Cortex.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

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
    IngestRequest,
    IngestResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint for load balancers and monitoring.
    
    No authentication required.
    """
    return HealthResponse()


@router.post("/ingest", response_model=IngestResponse, tags=["Ingest"])
async def ingest_session(
    request: IngestRequest,
    _api_key: ApiKeyDep,
    ingestion_service: IngestionServiceDep,
) -> IngestResponse:
    """
    Process a completed chat session into the knowledge graph.
    
    This endpoint:
    1. Validates the session meets minimum requirements
    2. Processes messages via Graphiti to extract entities/relationships
    3. Returns the updated user knowledge compilation
    
    Requires X-API-SECRET header for authentication.
    """
    return await ingestion_service.process_session(request)


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
