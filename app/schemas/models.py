from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Field, Tag


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

    jobId: str = Field(..., description="Convex queue table _id for async tracking")
    userId: str
    sessionId: str
    messages: list[IngestMessage]
    metadata: IngestMetadata


class IngestResponseMetadata(BaseModel):
    """Metadata about the ingestion processing."""

    model: str = Field(..., description="Gemini model used for ingestion")
    processing_time_ms: float | None = Field(default=None, description="Wall-clock time for Graphiti processing")
    nodes_extracted: int | None = Field(default=None, description="Number of entity nodes extracted")
    edges_extracted: int | None = Field(default=None, description="Number of entity edges extracted")
    episode_id: str | None = Field(default=None, description="UUID of the created episode")


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint (legacy sync flow)."""

    success: bool
    userKnowledgeCompilation: str | None = None
    metadata: IngestResponseMetadata | None = None
    error: str | None = None
    code: str | None = None


class CompilationMetadataResponse(BaseModel):
    """V2 hydration metadata for GraphRAG deduplication."""

    is_partial: bool
    total_estimated_tokens: int
    included_node_ids: list[str]
    included_edge_ids: list[str]
    included_episode_ids: list[str] = Field(default_factory=list)


class IngestAcceptedResponse(BaseModel):
    """202 response for POST /ingest (async fire-and-forget)."""

    jobId: str
    status: Literal["processing", "skipped"]
    userKnowledgeCompilation: str | None = None
    compilationMetadata: CompilationMetadataResponse | None = None


class GraphStatsResponse(BaseModel):
    """Total graph statistics for the user's knowledge graph."""

    entity_count: int = Field(..., description="Total Entity nodes in the graph")
    relationship_count: int = Field(..., description="Total valid RELATES_TO edges")
    total_chars: int = Field(default=0, description="Sum of all entity summaries + relationship facts in chars")


class IngestStatusResponse(BaseModel):
    """Response for GET /ingest/status/{job_id}."""

    jobId: str
    status: Literal["processing", "completed", "failed"]
    userKnowledgeCompilation: str | None = None
    compilationMetadata: CompilationMetadataResponse | None = None
    graphStats: GraphStatsResponse | None = None
    metadata: IngestResponseMetadata | None = None
    error: str | None = None
    code: str | None = None


# =============================================================================
# Chat Completions Models (OpenAI-compatible)
# =============================================================================


class TextContentPart(BaseModel):
    """A text content part in a multimodal message."""

    type: Literal["text"]
    text: str


class ImageUrlData(BaseModel):
    """Image URL data."""

    url: str


class ImageUrlContentPart(BaseModel):
    """An image_url content part in a multimodal message."""

    type: Literal["image_url"]
    image_url: ImageUrlData


ContentPart = Annotated[
    Annotated[TextContentPart, Tag("text")]
    | Annotated[ImageUrlContentPart, Tag("image_url")],
    Discriminator("type"),
]


class ChatMessage(BaseModel):
    """A message in the chat completion request."""

    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class ChatCompletionRequest(BaseModel):
    """Request body for the /v1/chat/completions endpoint."""

    messages: list[ChatMessage]
    model: str = Field(default="gemini-3-flash-preview", description="Model to use for completion")
    stream: bool = Field(default=True, description="Whether to stream the response")
    compilationMetadata: CompilationMetadataResponse | None = None
    user_id: str | None = Field(default=None, description="User/group ID for GraphRAG context retrieval")
    rag_usage_fields: dict[str, object] = Field(default_factory=dict, exclude=True)
    posthog_trace_id: str = Field(default="", exclude=True)


class ChatCompletionDelta(BaseModel):
    """Delta content in a streaming response chunk."""

    content: str | None = None
    role: str | None = None


class ChatCompletionChoice(BaseModel):
    """A choice in a streaming response chunk."""

    index: int
    delta: ChatCompletionDelta
    finish_reason: str | None = None


class UsageData(BaseModel):
    """Token usage statistics for a generation request (OpenAI-compatible)."""

    prompt_tokens: int = Field(..., description="Tokens in the input prompt")
    completion_tokens: int = Field(..., description="Tokens in the generated response")
    total_tokens: int = Field(..., description="Total tokens (prompt + completion)")
    thoughts_tokens: int | None = Field(default=None, description="Thinking tokens (Gemini 2.5+ models)")
    cached_tokens: int | None = Field(default=None, description="Tokens served from cache")
    rag_enabled: bool | None = Field(default=None, description="Whether GraphRAG context retrieval ran")
    rag_edges: int | None = Field(default=None, description="Episodic edges injected into the prompt")
    rag_nodes: int | None = Field(default=None, description="Entity nodes injected into the prompt")
    rag_search_ms: float | None = Field(default=None, description="Graphiti search latency in ms")
    rag_context_chars: int | None = Field(default=None, description="Characters of RAG context injected")


class ChatCompletionChunk(BaseModel):
    """A single chunk in a streaming chat completion response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageData | None = None


# =============================================================================
# Hydrate Endpoint Models (Debug/Read-only)
# =============================================================================


class HydrateRequest(BaseModel):
    """Request body for the /hydrate endpoint."""

    userId: str
    version: Literal["v1", "v2"] = "v1"


class HydrateResponse(BaseModel):
    """Response body for the /hydrate endpoint."""

    success: bool
    userKnowledgeCompilation: str | None = None
    compilationMetadata: CompilationMetadataResponse | None = None
    graphStats: GraphStatsResponse | None = None
    error: str | None = None
    code: str | None = None


# =============================================================================
# Graph Explorer Models (Memory Visualization & Correction)
# =============================================================================


class GraphNode(BaseModel):
    """A node in the knowledge graph visualization."""

    id: str = Field(..., description="Unique node identifier (entity name)")
    name: str = Field(..., description="Display name for the node")
    val: int = Field(..., description="Number of connections (controls node size)")
    summary: str = Field(..., description="Entity summary from Graphiti")


class GraphLink(BaseModel):
    """A link (edge) in the knowledge graph visualization."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    label: str = Field(..., description="Relationship type (e.g. RELATES_TO)")
    fact: str | None = Field(default=None, description="Fact associated with the relationship")


class GraphResponse(BaseModel):
    """Response body for the GET /v1/graph/{group_id} endpoint."""

    nodes: list[GraphNode]
    links: list[GraphLink]


class GraphCorrectionRequest(BaseModel):
    """Request body for the POST /v1/graph/correction endpoint."""

    group_id: str = Field(..., description="User/group ID whose memory to correct")
    correction_text: str = Field(..., description="Natural language correction to apply")


class GraphCorrectionResponse(BaseModel):
    """Response body for the POST /v1/graph/correction endpoint."""

    success: bool
    error: str | None = None
    code: str | None = None


# =============================================================================
# Health Check Models
# =============================================================================


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str = "ok"
    service: str = "synapse-cortex"


# =============================================================================
# Notion Export Models
# =============================================================================


class NotionExportRequest(BaseModel):
    """Request body for the POST /v1/notion/export endpoint."""

    userId: str
    notionToken: str = Field(..., description="Notion internal integration secret")
    pageName: str = Field(..., description="Name of the parent Notion page to export under")
    language: str = Field(default="English", description="Output language for all generated Notion content")


class NotionExportAcceptedResponse(BaseModel):
    """202 response for POST /v1/notion/export (async fire-and-forget)."""

    jobId: str
    status: Literal["processing"] = "processing"
    pageId: str = Field(..., description="Resolved Notion parent page ID")


class NotionExportProgress(BaseModel):
    """Current pipeline progress included in status polling responses."""

    currentStep: str
    categoriesDesigned: int | None = None
    entriesExtracted: int | None = None
    databasesCreated: int | None = None


class NotionExportResult(BaseModel):
    """Final result payload when the export completes successfully."""

    databaseIds: dict[str, str]
    summaryPageUrl: str | None = None
    categoriesCount: int
    entriesCount: int
    durationMs: float


class NotionExportStatusResponse(BaseModel):
    """Response for GET /v1/notion/export/status/{job_id}."""

    jobId: str
    status: Literal["processing", "completed", "failed"]
    progress: NotionExportProgress | None = None
    result: NotionExportResult | None = None
    error: str | None = None
    code: str | None = None


# =============================================================================
# Notion Correction Import Models
# =============================================================================


class NotionCorrectionRequest(BaseModel):
    """Request body for the POST /v1/notion/corrections endpoint."""

    userId: str
    notionToken: str = Field(..., description="Notion internal integration secret")
    pageName: str = Field(..., description="Name of the parent Notion page containing exported databases")
    language: str = Field(default="English", description="Language for extracted facts and summaries")


class NotionCorrectionAcceptedResponse(BaseModel):
    """202 response for POST /v1/notion/corrections (async fire-and-forget)."""

    jobId: str
    status: Literal["processing"] = "processing"
    pageId: str = Field(..., description="Resolved Notion parent page ID")


class NotionCorrectionProgress(BaseModel):
    """Current pipeline progress included in correction status polling responses."""

    currentStep: str
    databasesScanned: int | None = None
    correctionsFound: int | None = None
    correctionsApplied: int | None = None
    correctionsFailed: int | None = None


class NotionCorrectionResult(BaseModel):
    """Final result payload when the correction import completes successfully."""

    correctionsFound: int
    correctionsApplied: int
    correctionsFailed: int
    failedCorrections: list[dict] | None = None
    durationMs: float


class NotionCorrectionStatusResponse(BaseModel):
    """Response for GET /v1/notion/corrections/status/{job_id}."""

    jobId: str
    status: Literal["processing", "completed", "failed"]
    progress: NotionCorrectionProgress | None = None
    result: NotionCorrectionResult | None = None
    error: str | None = None
    code: str | None = None
