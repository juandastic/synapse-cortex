from dataclasses import dataclass, field


@dataclass
class CompilationMetadata:
    is_partial: bool
    total_estimated_tokens: int
    included_node_ids: list[str] = field(default_factory=list)
    included_edge_ids: list[str] = field(default_factory=list)
    included_episode_ids: list[str] = field(default_factory=list)
    compilation_revision: str | None = None
    compiler_model: str | None = None
    source_node_ids: list[str] = field(default_factory=list)
    source_edge_ids: list[str] = field(default_factory=list)
    citation_sources: dict[str, dict[str, list[str]]] = field(default_factory=dict)


@dataclass
class GraphStats:
    """Total graph-wide counts + content size (unfiltered by compilation budget)."""

    entity_count: int
    relationship_count: int
    total_chars: int = 0


@dataclass
class HydrationResult:
    compilation_text: str
    metadata: CompilationMetadata | None = None
    graph_stats: GraphStats | None = None
    provider_cache_name: str | None = None
    provider_cache_model: str | None = None
    provider_cache_expires_at: float = 0
