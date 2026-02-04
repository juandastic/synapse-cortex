from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# Ingest Endpoint Models
# =============================================================================


class IngestMessage(BaseModel):
    """A single message from a chat session."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: int = Field(..., description="Unix timestamp in milliseconds")


class IngestMetadata(BaseModel):
    """Metadata about the chat session."""

    sessionStartedAt: int = Field(..., description="Unix timestamp in milliseconds")
    sessionEndedAt: int = Field(..., description="Unix timestamp in milliseconds")
    messageCount: int


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""

    userId: str
    sessionId: str
    messages: list[IngestMessage]
    metadata: IngestMetadata


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""

    success: bool
    userKnowledgeCompilation: str | None = None
    error: str | None = None
    code: str | None = None


# =============================================================================
# Chat Completions Models (OpenAI-compatible)
# =============================================================================


class ChatMessage(BaseModel):
    """A message in the chat completion request."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Request body for the /v1/chat/completions endpoint."""

    messages: list[ChatMessage]
    model: str = Field(default="gemini-3-flash-preview", description="Model to use for completion")
    stream: bool = Field(default=True, description="Whether to stream the response")


class ChatCompletionDelta(BaseModel):
    """Delta content in a streaming response chunk."""

    content: str | None = None
    role: str | None = None


class ChatCompletionChoice(BaseModel):
    """A choice in a streaming response chunk."""

    index: int
    delta: ChatCompletionDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """A single chunk in a streaming chat completion response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


# =============================================================================
# Hydrate Endpoint Models (Debug/Read-only)
# =============================================================================


class HydrateRequest(BaseModel):
    """Request body for the /hydrate endpoint."""

    userId: str


class HydrateResponse(BaseModel):
    """Response body for the /hydrate endpoint."""

    success: bool
    userKnowledgeCompilation: str | None = None
    error: str | None = None
    code: str | None = None


# =============================================================================
# Health Check Models
# =============================================================================


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str = "ok"
    service: str = "synapse-cortex"
