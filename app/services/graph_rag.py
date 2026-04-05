"""
GraphRAG - Graph-based Retrieval-Augmented Generation.

Performs a lightweight hybrid search over edges *and* nodes on every chat
turn to retrieve long-tail episodic memories and entity details not covered
by the static Waterfill base prompt, deduplicates against already-included
items, and returns a formatted context block ready for injection.

All functions are pure (no mutation of input arguments).
"""

import html
import logging
import re
import time
from dataclasses import dataclass

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_RRF,
)

from app.core.posthog import capture_span, new_trace_id, posthog_user_context
from app.schemas.models import ChatCompletionRequest, ChatMessage

logger = logging.getLogger(__name__)

_QUERY_TAIL_MESSAGES = 3
_RAG_HEADER = (
    "\n\n### RELEVANT EPISODIC MEMORY FOR THIS TURN ###\n"
    "(The following facts and entities were retrieved from the user's "
    "graph memory based on the current conversation context. "
    "Use them if relevant.)\n\n"
)

EDGE_AND_NODE_HYBRID_SEARCH_RRF = SearchConfig(
    edge_config=EDGE_HYBRID_SEARCH_RRF.edge_config,
    node_config=NODE_HYBRID_SEARCH_RRF.node_config,
    episode_config=None,
    community_config=None,
    limit=10,
)


@dataclass(frozen=True)
class GraphRagResult:
    """Immutable return value from the GraphRAG retrieval pipeline.

    Carries the formatted context block and all telemetry stats the caller
    needs to record on the OTel span.  Does *not* carry or modify messages.
    """

    context_block: str
    raw_edges_count: int
    deduped_edges_count: int
    injected_edges_count: int
    raw_nodes_count: int
    deduped_nodes_count: int
    injected_nodes_count: int
    search_duration_ms: float
    total_duration_ms: float
    query_chars: int

    @property
    def context_block_chars(self) -> int:
        return len(self.context_block)

    @property
    def has_context(self) -> bool:
        return self.injected_edges_count > 0 or self.injected_nodes_count > 0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _extract_text(message: ChatMessage) -> str:
    """Extract plain text from a ChatMessage regardless of content type."""
    if isinstance(message.content, str):
        return message.content
    return " ".join(
        part.text for part in message.content if hasattr(part, "text")
    )


def build_search_query(messages: list[ChatMessage]) -> str:
    """Concatenate the last N user/assistant messages into a search query.

    Using only the very last message (which could be "Why?") produces poor
    recall.  Taking a short window gives the embedding + BM25 engines enough
    signal to match relevant edges and nodes.
    """
    tail: list[str] = []
    for msg in reversed(messages):
        if msg.role == "system":
            continue
        tail.append(_extract_text(msg))
        if len(tail) >= _QUERY_TAIL_MESSAGES:
            break
    tail.reverse()
    return "\n".join(tail)


_MULTILINE_WS = re.compile(r"\s*\n\s*")


def _clean_text(text: str) -> str:
    """Decode HTML entities and collapse stray newlines into a single space."""
    text = html.unescape(text)
    return _MULTILINE_WS.sub(" ", text).strip()


def _format_edge_name(name: str) -> str:
    """Convert edge name to readable verb (e.g., WORKS_WITH -> works with)."""
    if not name:
        return "relates to"
    return name.replace("_", " ").lower()


def format_edges(edges: list[EntityEdge]) -> str:
    """Format edges into compact readable lines matching hydration style."""
    lines: list[str] = []
    for edge in edges:
        verb = _format_edge_name(edge.name)
        fact = _clean_text(edge.fact)
        line = f"- ({verb}) {fact}"
        if edge.valid_at:
            line += f" [since: {edge.valid_at.strftime('%Y-%m-%d')}]"
        lines.append(line)
    return "\n".join(lines)


def format_nodes(nodes: list[EntityNode]) -> str:
    """Format nodes into readable lines matching hydration style."""
    lines: list[str] = []
    for node in nodes:
        summary = _clean_text(node.summary)
        lines.append(f"- **{node.name}**: {summary}")
    return "\n".join(lines)


def deduplicate_edges(
    edges: list[EntityEdge],
    included_ids: set[str],
) -> list[EntityEdge]:
    """Return only edges whose UUID is not already in the base prompt."""
    return [e for e in edges if e.uuid not in included_ids]


def deduplicate_nodes(
    nodes: list[EntityNode],
    included_ids: set[str],
) -> list[EntityNode]:
    """Return only nodes whose UUID is not already in the base prompt."""
    return [n for n in nodes if n.uuid not in included_ids]


def build_context_block(
    edges: list[EntityEdge],
    nodes: list[EntityNode],
) -> str:
    """Build a single context block string from edges and nodes."""
    sections: list[str] = []
    if edges:
        sections.append(format_edges(edges))
    if nodes:
        sections.append(format_nodes(nodes))
    return "\n".join(sections)


def build_messages_with_context(
    messages: list[ChatMessage],
    context_block: str,
) -> list[ChatMessage]:
    """Return a new message list with the RAG context appended to the system message.

    Gemini conversion in GenerationService only reads the *first* system
    message and silently skips the rest, so we append to the existing one
    rather than inserting a second system message.

    The original list and its messages are never modified.
    """
    augmented = _RAG_HEADER + context_block
    found_system = False
    result: list[ChatMessage] = []

    for msg in messages:
        if not found_system and msg.role == "system" and isinstance(msg.content, str):
            result.append(ChatMessage(
                role="system",
                content=msg.content + augmented,
            ))
            found_system = True
        else:
            result.append(msg)

    if not found_system:
        result.insert(0, ChatMessage(role="system", content=augmented))

    return result


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

async def retrieve_graph_rag_context(
    graphiti: Graphiti,
    messages: list[ChatMessage],
    user_id: str,
    included_edge_ids: set[str],
    included_node_ids: set[str],
    posthog_trace_id: str = "",
) -> GraphRagResult:
    """Run the full GraphRAG pipeline: query -> search -> dedup -> format.

    This is a pure async function: it reads *messages* to build a query but
    never modifies them.  The caller is responsible for assembling the final
    message list via ``build_messages_with_context``.
    """
    start_total = time.monotonic()

    query = build_search_query(messages)

    start_search = time.monotonic()
    with posthog_user_context(user_id, posthog_trace_id):
        results = await graphiti.search_(
            query,
            config=EDGE_AND_NODE_HYBRID_SEARCH_RRF,
            group_ids=[user_id],
        )
    search_ms = (time.monotonic() - start_search) * 1000

    new_edges = deduplicate_edges(results.edges, included_edge_ids)
    new_nodes = deduplicate_nodes(results.nodes, included_node_ids)
    context_block = build_context_block(new_edges, new_nodes)
    total_ms = (time.monotonic() - start_total) * 1000

    deduped_edges = len(results.edges) - len(new_edges)
    deduped_nodes = len(results.nodes) - len(new_nodes)
    total_injected = len(new_edges) + len(new_nodes)

    logger.info(
        "GraphRAG context: %d injected (%d edges, %d nodes), "
        "%d raw (%d edges, %d nodes), "
        "%d deduped (%d edges, %d nodes) in %.1fms",
        total_injected, len(new_edges), len(new_nodes),
        len(results.edges) + len(results.nodes), len(results.edges), len(results.nodes),
        deduped_edges + deduped_nodes, deduped_edges, deduped_nodes,
        total_ms,
    )
    if new_edges:
        logger.debug("GraphRAG edges: %s", [(e.uuid, e.fact) for e in new_edges])
    if new_nodes:
        logger.debug("GraphRAG nodes: %s", [(n.uuid, n.name) for n in new_nodes])

    # PostHog LLM Analytics — uses same trace_id as the chat completion
    capture_span(
        user_id,
        posthog_trace_id or new_trace_id(),
        name="graphiti.search",
        input_data=[{"role": "user", "content": query[:2000]}],
        output_data=[{"role": "assistant", "content": context_block[:3000] or "(no relevant context found)"}],
        duration_ms=total_ms,
        properties={
            "pipeline": "graph_rag",
            "raw_edges": len(results.edges),
            "raw_nodes": len(results.nodes),
            "deduped_edges": deduped_edges,
            "deduped_nodes": deduped_nodes,
            "injected_edges": len(new_edges),
            "injected_nodes": len(new_nodes),
        },
    )

    return GraphRagResult(
        context_block=context_block,
        raw_edges_count=len(results.edges),
        deduped_edges_count=deduped_edges,
        injected_edges_count=len(new_edges),
        raw_nodes_count=len(results.nodes),
        deduped_nodes_count=deduped_nodes,
        injected_nodes_count=len(new_nodes),
        search_duration_ms=round(search_ms, 2),
        total_duration_ms=round(total_ms, 2),
        query_chars=len(query),
    )


# ---------------------------------------------------------------------------
# Orchestration: gating, execution, and telemetry mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphRagOutcome:
    """Encapsulates the result of the GraphRAG gating + retrieval attempt."""

    enabled: bool
    skip_reason: str = ""
    result: GraphRagResult | None = None


def get_rag_skip_reason(request: ChatCompletionRequest) -> str:
    """Return the reason GraphRAG should be skipped, or empty string if it should run."""
    if not request.user_id:
        return "no_user_id"
    if not request.compilationMetadata:
        return "no_compilation_metadata"
    if not request.compilationMetadata.is_partial:
        return "graph_fully_loaded"
    return ""


async def maybe_run_graph_rag(
    request: ChatCompletionRequest,
    graphiti: Graphiti,
) -> GraphRagOutcome:
    """Gate-check and, if appropriate, execute GraphRAG retrieval.

    On success the request's messages are replaced with a new list that
    includes the RAG context block (no mutation of the originals).
    """
    skip_reason = get_rag_skip_reason(request)
    if skip_reason:
        return GraphRagOutcome(enabled=False, skip_reason=skip_reason)

    assert request.user_id is not None
    assert request.compilationMetadata is not None

    try:
        rag_result = await retrieve_graph_rag_context(
            graphiti=graphiti,
            messages=request.messages,
            user_id=request.user_id,
            included_edge_ids=set(request.compilationMetadata.included_edge_ids),
            included_node_ids=set(request.compilationMetadata.included_node_ids),
            posthog_trace_id=request.posthog_trace_id,
        )
        if rag_result.has_context:
            request.messages = build_messages_with_context(
                request.messages, rag_result.context_block,
            )
        return GraphRagOutcome(enabled=True, result=rag_result)
    except Exception:
        logger.exception("GraphRAG context retrieval failed, proceeding without it")
        return GraphRagOutcome(enabled=False, skip_reason="error")


def rag_outcome_to_span_attrs(outcome: GraphRagOutcome) -> dict[str, object]:
    """Convert a GraphRagOutcome into OTel span attributes (``rag.*`` namespace)."""
    attrs: dict[str, object] = {"rag.enabled": outcome.enabled}
    if not outcome.enabled:
        attrs["rag.skipped_reason"] = outcome.skip_reason
        return attrs

    if outcome.result is not None:
        r = outcome.result
        attrs.update({
            "rag.search_duration_ms": r.search_duration_ms,
            "rag.total_duration_ms": r.total_duration_ms,
            "rag.raw_edges_count": r.raw_edges_count,
            "rag.deduped_edges_count": r.deduped_edges_count,
            "rag.injected_edges_count": r.injected_edges_count,
            "rag.raw_nodes_count": r.raw_nodes_count,
            "rag.deduped_nodes_count": r.deduped_nodes_count,
            "rag.injected_nodes_count": r.injected_nodes_count,
            "rag.query_chars": r.query_chars,
            "rag.context_block_chars": r.context_block_chars,
        })
    return attrs


def rag_outcome_to_usage_fields(outcome: GraphRagOutcome) -> dict[str, object]:
    """Return the RAG fields to merge into ``UsageData`` on the final SSE chunk."""
    if not outcome.enabled or outcome.result is None:
        return {"rag_enabled": False}
    r = outcome.result
    return {
        "rag_enabled": True,
        "rag_edges": r.injected_edges_count,
        "rag_nodes": r.injected_nodes_count,
        "rag_search_ms": r.search_duration_ms,
        "rag_context_chars": r.context_block_chars,
    }
