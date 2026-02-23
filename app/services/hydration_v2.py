"""
Hydration V2 Engine - Budget-aware compilation with cascading waterfill allocation.

Maximizes context within a character limit using priority-based edge selection
and in-memory hub classification from node degree data.
"""

import time
from dataclasses import dataclass

from neo4j import AsyncDriver
from opentelemetry import trace

from app.core.observability import (
    anonymize_id,
    classify_error,
    mark_span_error,
    mark_span_success,
    set_span_attributes,
)
from app.services.hydration_result import CompilationMetadata, HydrationResult

DEFAULT_CHAR_LIMIT = 120_000
NODE_BUDGET_RATIO = 0.40
# Nodes rarely consume their full 40%; leftover rolls into the edge budget
EDGE_BUDGET_RATIO = 0.60
# Entities in the top 30% by degree are considered structural hubs
HUB_PERCENTILE = 0.70

FETCH_NODES_QUERY = """
MATCH (n:Entity)-[r:RELATES_TO]-(other:Entity)
WHERE n <> other
  AND n.group_id = $group_id
  AND n.summary IS NOT NULL
  AND n.summary <> ""
WITH n, count(r) AS degree
WHERE degree >= $min_degree
RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, degree
ORDER BY degree DESC
"""

FETCH_EDGES_QUERY = """
MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
WHERE r.group_id = $group_id
  AND (r.invalid_at IS NULL OR r.invalid_at > datetime())
  AND NOT 'Episode' IN labels(source)
  AND NOT 'Episode' IN labels(target)
RETURN
    r.uuid AS uuid,
    source.name AS source_name,
    target.name AS target_name,
    r.name AS relation_name,
    r.fact AS fact,
    r.valid_at AS valid_at,
    r.invalid_at AS invalid_at,
    r.created_at AS created_at
ORDER BY r.created_at DESC
"""


def _compact_date(value: object) -> str:
    """Extract YYYY-MM-DD from Neo4j DateTime or ISO string."""
    return str(value)[:10]


@dataclass
class NodeRecord:
    uuid: str
    name: str
    summary: str
    degree: int
    formatted: str

    @property
    def char_cost(self) -> int:
        return len(self.formatted)


@dataclass
class EdgeRecord:
    uuid: str
    source_name: str
    target_name: str
    formatted: str
    recency_key: object  # datetime used for sorting; opaque to avoid parsing

    @property
    def char_cost(self) -> int:
        return len(self.formatted)


class HydrationV2Engine:
    def __init__(
        self,
        driver: AsyncDriver,
        min_degree: int,
        char_limit: int = DEFAULT_CHAR_LIMIT,
    ):
        self.driver = driver
        self.min_degree = min_degree
        self.char_limit = char_limit

    async def build(self, user_id: str) -> HydrationResult:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "hydration_v2.build",
            attributes={
                "hydrate.user_id": anonymize_id(user_id),
                "hydrate.version": "v2",
                "hydrate.char_limit": self.char_limit,
                "hydrate.min_degree": self.min_degree,
            },
        ) as span:
            start = time.monotonic()
            try:
                nodes = await self._fetch_nodes(user_id)
                node_degree_map = {n.name: n.degree for n in nodes}

                edges = await self._fetch_edges(user_id)

                if not nodes and not edges:
                    set_span_attributes(span, {
                        "hydrate.nodes_count": 0,
                        "hydrate.edges_count": 0,
                        "duration_ms": round((time.monotonic() - start) * 1000, 2),
                    })
                    mark_span_success(span)
                    return HydrationResult(compilation_text="")

                total_chars = (
                    sum(n.char_cost for n in nodes)
                    + sum(e.char_cost for e in edges)
                )

                if total_chars <= self.char_limit:
                    result = self._build_fast_path(nodes, edges, total_chars)
                else:
                    result = self._build_with_budget(nodes, edges, node_degree_map)

                set_span_attributes(span, {
                    "hydrate.nodes_count": len(nodes),
                    "hydrate.edges_count": len(edges),
                    "hydrate.is_partial": result.metadata.is_partial if result.metadata else False,
                    "hydrate.compilation_size_chars": len(result.compilation_text),
                    "hydrate.total_estimated_tokens": result.metadata.total_estimated_tokens if result.metadata else 0,
                    "duration_ms": round((time.monotonic() - start) * 1000, 2),
                })
                mark_span_success(span)
                return result

            except Exception as e:
                category, code = classify_error(e)
                mark_span_error(span, e, category=category, code=code)
                raise

    # ── Data Fetching ────────────────────────────────────────────────────

    async def _fetch_nodes(self, group_id: str) -> list[NodeRecord]:
        async with self.driver.session() as session:
            result = await session.run(
                FETCH_NODES_QUERY,
                group_id=group_id,
                min_degree=self.min_degree,
            )
            records = await result.data()

        return [
            NodeRecord(
                uuid=r["uuid"],
                name=r["name"],
                summary=r["summary"],
                degree=r["degree"],
                formatted=f"- **{r['name']}**: {r['summary']}",
            )
            for r in records
            if r.get("name") and r.get("summary")
        ]

    async def _fetch_edges(self, group_id: str) -> list[EdgeRecord]:
        async with self.driver.session() as session:
            result = await session.run(FETCH_EDGES_QUERY, group_id=group_id)
            records = await result.data()

        edges = []
        for r in records:
            source = r.get("source_name", "")
            target = r.get("target_name", "")
            if not source or not target:
                continue

            verb = (r.get("relation_name") or "").replace("_", " ").lower() or "relates to"
            line = f"- {source} {verb} {target}"

            fact = r.get("fact")
            if fact:
                line += f': "{fact}"'

            temporal_parts = []
            if r.get("valid_at"):
                temporal_parts.append(f"since: {_compact_date(r['valid_at'])}")
            if r.get("invalid_at"):
                temporal_parts.append(f"until: {_compact_date(r['invalid_at'])}")
            if temporal_parts:
                line += f" [{', '.join(temporal_parts)}]"

            edges.append(EdgeRecord(
                uuid=r["uuid"],
                source_name=source,
                target_name=target,
                formatted=line,
                recency_key=r.get("valid_at") or r.get("created_at"),
            ))

        return edges

    # ── Fast Path (everything fits) ──────────────────────────────────────

    def _build_fast_path(
        self,
        nodes: list[NodeRecord],
        edges: list[EdgeRecord],
        total_chars: int,
    ) -> HydrationResult:
        compilation = self._format_compilation(
            [n.formatted for n in nodes],
            [e.formatted for e in edges],
        )
        return HydrationResult(
            compilation_text=compilation,
            metadata=CompilationMetadata(
                is_partial=False,
                total_estimated_tokens=total_chars // 4,
                included_node_ids=[n.uuid for n in nodes],
                included_edge_ids=[e.uuid for e in edges],
            ),
        )

    # ── Waterfill Allocation (budget-constrained) ────────────────────────

    def _build_with_budget(
        self,
        nodes: list[NodeRecord],
        edges: list[EdgeRecord],
        node_degree_map: dict[str, int],
    ) -> HydrationResult:
        node_budget = int(self.char_limit * NODE_BUDGET_RATIO)

        included_nodes = self._select_nodes_within_budget(nodes, node_budget)
        used_node_chars = sum(n.char_cost for n in included_nodes)
        rollover = node_budget - used_node_chars

        edge_budget = int(self.char_limit * EDGE_BUDGET_RATIO) + rollover
        hub_threshold = self._compute_hub_threshold(node_degree_map)
        included_edges = self._select_edges_within_budget(
            edges, edge_budget, node_degree_map, hub_threshold,
        )

        total_chars = used_node_chars + sum(e.char_cost for e in included_edges)
        compilation = self._format_compilation(
            [n.formatted for n in included_nodes],
            [e.formatted for e in included_edges],
        )
        return HydrationResult(
            compilation_text=compilation,
            metadata=CompilationMetadata(
                is_partial=True,
                total_estimated_tokens=total_chars // 4,
                included_node_ids=[n.uuid for n in included_nodes],
                included_edge_ids=[e.uuid for e in included_edges],
            ),
        )

    @staticmethod
    def _select_nodes_within_budget(
        nodes: list[NodeRecord],
        budget: int,
    ) -> list[NodeRecord]:
        selected: list[NodeRecord] = []
        remaining = budget
        for node in nodes:  # already sorted by degree DESC from Cypher
            if node.char_cost > remaining:
                break
            selected.append(node)
            remaining -= node.char_cost
        return selected

    @staticmethod
    def _compute_hub_threshold(node_degree_map: dict[str, int]) -> int:
        if not node_degree_map:
            return 0
        degrees = sorted(node_degree_map.values())
        cutoff_index = int(len(degrees) * HUB_PERCENTILE)
        return degrees[min(cutoff_index, len(degrees) - 1)]

    @staticmethod
    def _select_edges_within_budget(
        edges: list[EdgeRecord],
        budget: int,
        node_degree_map: dict[str, int],
        hub_threshold: int,
    ) -> list[EdgeRecord]:
        def is_hub(name: str) -> bool:
            return node_degree_map.get(name, 0) >= hub_threshold

        hub_to_hub: list[EdgeRecord] = []
        hub_adjacent: list[EdgeRecord] = []
        long_tail: list[EdgeRecord] = []

        for edge in edges:
            source_is_hub = is_hub(edge.source_name)
            target_is_hub = is_hub(edge.target_name)

            if source_is_hub and target_is_hub:
                hub_to_hub.append(edge)
            elif source_is_hub or target_is_hub:
                hub_adjacent.append(edge)
            else:
                long_tail.append(edge)

        # P2 and P3 sorted by recency (P1 enters almost entirely by default)
        hub_adjacent.sort(key=lambda e: e.recency_key or "", reverse=True)
        long_tail.sort(key=lambda e: e.recency_key or "", reverse=True)

        selected: list[EdgeRecord] = []
        remaining = budget

        for tier in (hub_to_hub, hub_adjacent, long_tail):
            for edge in tier:
                if edge.char_cost > remaining:
                    continue
                selected.append(edge)
                remaining -= edge.char_cost

        return selected

    # ── Formatting (mirrors V1 structure for LLM consistency) ────────────

    @staticmethod
    def _format_compilation(
        node_lines: list[str],
        edge_lines: list[str],
    ) -> str:
        sections = []

        if node_lines:
            sections.append(
                "#### 1. CONCEPTUAL DEFINITIONS & IDENTITY ####\n"
                "# (Understanding what these concepts mean specifically for this user)\n"
                + "\n".join(node_lines)
            )

        if edge_lines:
            sections.append(
                "#### 2. RELATIONAL DYNAMICS & CAUSALITY ####\n"
                "# (How these concepts interact and evolve over time)\n"
                + "\n".join(edge_lines)
            )

        if not sections:
            return ""

        content = "\n\n".join(sections)

        total_chars = sum(len(l) for l in node_lines) + sum(len(l) for l in edge_lines)
        est_tokens = total_chars // 4
        stats = (
            f"\n\n### STATS ###\n"
            f"# Definitions: {len(node_lines)} | Relations: {len(edge_lines)} | Est. Tokens: ~{est_tokens}"
        )

        return content + stats
