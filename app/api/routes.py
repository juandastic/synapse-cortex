"""
API Routes - Endpoint definitions for Synapse Cortex.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    ApiKeyDep,
    GenerationServiceDep,
    HydrationServiceDep,
    IngestionServiceDep,
)
from app.schemas.models import (
    ChatCompletionRequest,
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
