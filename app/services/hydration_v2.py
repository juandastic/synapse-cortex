"""
Hydration V2 Engine - Budget-aware compilation with cascading waterfill allocation.

Maximizes context within a character limit using priority-based edge selection,
in-memory hub classification from node degree data, and episodic memory retrieval.

Compilation is organized into four sections:
  1. SESSION HISTORY — chronological episode summaries
  2. ACTIVE CONTEXT — recently updated facts and relationships
  3. CORE IDENTITY & DEFINITIONS — stable entities and hub relationships
  4. BEHAVIORAL PATTERNS & DYNAMICS — long-tail behavioral edges
"""

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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

# 4-tier budget allocation (rollover cascades: episode -> active -> identity -> dynamics)
EPISODE_BUDGET_RATIO = 0.08   # ~9.6K chars (episodes are small, ~3K typical)
ACTIVE_BUDGET_RATIO = 0.12    # ~14.4K chars (recent context, changes frequently)
IDENTITY_BUDGET_RATIO = 0.45  # ~54K chars (nodes + stable hub edges — the bulk)
DYNAMICS_BUDGET_RATIO = 0.35  # ~42K chars (long-tail behavioral edges)

# Entities in the top 30% by degree are considered structural hubs
HUB_PERCENTILE = 0.70

# Edges created within this window are considered "recent" (Active Context section)
RECENCY_WINDOW_DAYS = 14

# Fallback: max chars of user messages to extract when no summary exists
EPISODE_CONTENT_FALLBACK_LIMIT = 200

_USER_PREFIX = "User: "

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

FETCH_EPISODES_QUERY = """
MATCH (e:Episodic)
WHERE e.group_id = $group_id
  AND e.source = 'message'
RETURN e.uuid AS uuid,
       e.name AS name,
       e.valid_at AS valid_at,
       e.summary AS summary,
       e.content AS content
ORDER BY e.valid_at DESC
LIMIT $limit
"""


def _compact_date(value: object) -> str:
    """Extract YYYY-MM-DD from Neo4j DateTime or ISO string."""
    return str(value)[:10]


def _extract_user_lines(content: str, limit: int) -> str:
    """Extract user messages from raw episode content as a fallback summary.

    Episode content is formatted as "User: ...\n\nAssistant: ..." blocks.
    We only keep the user parts (the valuable signal) and join them with " | ".
    """
    parts: list[str] = []
    total = 0
    for block in content.split("\n\n"):
        block = block.strip()
        if block.startswith(_USER_PREFIX):
            text = block[len(_USER_PREFIX):].strip()
            if not text:
                continue
            if total + len(text) > limit:
                remaining = limit - total
                if remaining > 20:
                    parts.append(text[:remaining] + "...")
                break
            parts.append(text)
            total += len(text)
    return " | ".join(parts) if parts else content[:limit] + "..."


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


@dataclass
class EpisodeRecord:
    uuids: list[str]  # all episode UUIDs merged into this day group
    name: str
    valid_at: object  # datetime
    summary: str
    formatted: str

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
                # Fetch all three data sources in parallel
                episodes, nodes, edges = await asyncio.gather(
                    self._fetch_episodes(user_id),
                    self._fetch_nodes(user_id),
                    self._fetch_edges(user_id),
                )
                node_degree_map = {n.name: n.degree for n in nodes}

                if not nodes and not edges and not episodes:
                    set_span_attributes(span, {
                        "hydrate.episodes_count": 0,
                        "hydrate.nodes_count": 0,
                        "hydrate.edges_count": 0,
                        "duration_ms": round((time.monotonic() - start) * 1000, 2),
                    })
                    mark_span_success(span)
                    return HydrationResult(compilation_text="")

                total_chars = (
                    sum(ep.char_cost for ep in episodes)
                    + sum(n.char_cost for n in nodes)
                    + sum(e.char_cost for e in edges)
                )

                if total_chars <= self.char_limit:
                    result = self._build_fast_path(episodes, nodes, edges, total_chars)
                else:
                    result = self._build_with_budget(episodes, nodes, edges, node_degree_map)

                set_span_attributes(span, {
                    "hydrate.episodes_count": len(episodes),
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

    async def _fetch_episodes(self, group_id: str, limit: int = 15) -> list[EpisodeRecord]:
        """Fetch recent episodes and group them by day.

        Multiple sessions on the same date are merged into a single entry
        so the timeline reads as one line per day, not one per session.
        """
        async with self.driver.session() as session:
            result = await session.run(
                FETCH_EPISODES_QUERY, group_id=group_id, limit=limit,
            )
            records = await result.data()

        # Group raw episode summaries by date (YYYY-MM-DD)
        days: OrderedDict[str, list[str]] = OrderedDict()
        uuids_by_day: OrderedDict[str, list[str]] = OrderedDict()
        first_valid_at: dict[str, object] = {}

        for r in records:
            summary = r.get("summary") or ""
            if not summary and r.get("content"):
                summary = _extract_user_lines(
                    r["content"], EPISODE_CONTENT_FALLBACK_LIMIT,
                )
            if not summary:
                continue
            date_str = _compact_date(r["valid_at"]) if r.get("valid_at") else "unknown"
            days.setdefault(date_str, []).append(summary)
            uuids_by_day.setdefault(date_str, []).append(r["uuid"])
            if date_str not in first_valid_at:
                first_valid_at[date_str] = r.get("valid_at")

        # Build one EpisodeRecord per day
        episodes = []
        for date_str, summaries in days.items():
            merged = " | ".join(summaries)
            formatted = f"- [{date_str}] {merged}"
            episodes.append(EpisodeRecord(
                uuids=uuids_by_day[date_str],
                name=date_str,
                valid_at=first_valid_at.get(date_str),
                summary=merged,
                formatted=formatted,
            ))
        return episodes

    # ── Fast Path (everything fits) ──────────────────────────────────────

    def _build_fast_path(
        self,
        episodes: list[EpisodeRecord],
        nodes: list[NodeRecord],
        edges: list[EdgeRecord],
        total_chars: int,
    ) -> HydrationResult:
        recent_edges, stable_edges = self._partition_edges_by_recency(edges)
        node_degree_map = {n.name: n.degree for n in nodes}
        hub_threshold = self._compute_hub_threshold(node_degree_map)
        identity_edges, dynamics_edges = self._partition_stable_edges(
            stable_edges, node_degree_map, hub_threshold,
        )

        compilation = self._format_compilation(
            episode_lines=[ep.formatted for ep in episodes],
            active_lines=[e.formatted for e in recent_edges],
            node_lines=[n.formatted for n in nodes],
            identity_edge_lines=[e.formatted for e in identity_edges],
            dynamics_lines=[e.formatted for e in dynamics_edges],
        )
        return HydrationResult(
            compilation_text=compilation,
            metadata=CompilationMetadata(
                is_partial=False,
                total_estimated_tokens=total_chars // 4,
                included_node_ids=[n.uuid for n in nodes],
                included_edge_ids=[e.uuid for e in edges],
                included_episode_ids=[uid for ep in episodes for uid in ep.uuids],
            ),
        )

    # ── Waterfill Allocation (budget-constrained) ────────────────────────

    def _build_with_budget(
        self,
        episodes: list[EpisodeRecord],
        nodes: list[NodeRecord],
        edges: list[EdgeRecord],
        node_degree_map: dict[str, int],
    ) -> HydrationResult:
        # Tier 1: Episodes
        episode_budget = int(self.char_limit * EPISODE_BUDGET_RATIO)
        included_episodes = self._select_within_budget(episodes, episode_budget)
        ep_used = sum(ep.char_cost for ep in included_episodes)
        rollover = episode_budget - ep_used

        # Tier 2: Active Context (recent edges)
        recent_edges, stable_edges = self._partition_edges_by_recency(edges)
        active_budget = int(self.char_limit * ACTIVE_BUDGET_RATIO) + rollover
        included_active = self._select_within_budget(recent_edges, active_budget)
        active_used = sum(e.char_cost for e in included_active)
        rollover = active_budget - active_used

        # Tier 3: Core Identity (nodes + stable hub edges)
        identity_budget = int(self.char_limit * IDENTITY_BUDGET_RATIO) + rollover
        # Nodes get first pick within the identity budget
        included_nodes = self._select_nodes_within_budget(nodes, identity_budget)
        nodes_used = sum(n.char_cost for n in included_nodes)
        identity_edge_budget = identity_budget - nodes_used

        hub_threshold = self._compute_hub_threshold(node_degree_map)
        identity_edges, dynamics_edges = self._partition_stable_edges(
            stable_edges, node_degree_map, hub_threshold,
        )
        included_identity_edges = self._select_within_budget(
            identity_edges, identity_edge_budget,
        )
        identity_edges_used = sum(e.char_cost for e in included_identity_edges)
        rollover = identity_edge_budget - identity_edges_used

        # Tier 4: Behavioral Patterns & Dynamics (long-tail edges)
        dynamics_budget = int(self.char_limit * DYNAMICS_BUDGET_RATIO) + rollover
        included_dynamics = self._select_within_budget(dynamics_edges, dynamics_budget)

        total_chars = (
            ep_used + active_used + nodes_used
            + identity_edges_used
            + sum(e.char_cost for e in included_dynamics)
        )

        compilation = self._format_compilation(
            episode_lines=[ep.formatted for ep in included_episodes],
            active_lines=[e.formatted for e in included_active],
            node_lines=[n.formatted for n in included_nodes],
            identity_edge_lines=[e.formatted for e in included_identity_edges],
            dynamics_lines=[e.formatted for e in included_dynamics],
        )

        all_included_edges = included_active + included_identity_edges + included_dynamics
        return HydrationResult(
            compilation_text=compilation,
            metadata=CompilationMetadata(
                is_partial=True,
                total_estimated_tokens=total_chars // 4,
                included_node_ids=[n.uuid for n in included_nodes],
                included_edge_ids=[e.uuid for e in all_included_edges],
                included_episode_ids=[uid for ep in included_episodes for uid in ep.uuids],
            ),
        )

    @staticmethod
    def _select_within_budget(items: list, budget: int) -> list:
        """Generic greedy selector: pick items in order until budget runs out."""
        selected = []
        remaining = budget
        for item in items:
            if item.char_cost > remaining:
                continue
            selected.append(item)
            remaining -= item.char_cost
        return selected

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
    def _partition_edges_by_recency(
        edges: list[EdgeRecord],
    ) -> tuple[list[EdgeRecord], list[EdgeRecord]]:
        """Split edges into recent (Active Context) and stable (older) groups."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=RECENCY_WINDOW_DAYS)
        recent: list[EdgeRecord] = []
        stable: list[EdgeRecord] = []
        for edge in edges:
            try:
                key = edge.recency_key
                # Neo4j DateTime objects have .to_native() or can be compared as strings
                if key and str(key)[:10] >= cutoff.strftime("%Y-%m-%d"):
                    recent.append(edge)
                else:
                    stable.append(edge)
            except (TypeError, ValueError):
                stable.append(edge)
        return recent, stable

    @staticmethod
    def _partition_stable_edges(
        stable_edges: list[EdgeRecord],
        node_degree_map: dict[str, int],
        hub_threshold: int,
    ) -> tuple[list[EdgeRecord], list[EdgeRecord]]:
        """Split stable edges into identity (hub-connected) and dynamics (long-tail).

        Identity edges: at least one endpoint is a hub (core relationships).
        Dynamics edges: neither endpoint is a hub (behavioral patterns, long-tail).
        """
        def is_hub(name: str) -> bool:
            return node_degree_map.get(name, 0) >= hub_threshold

        identity: list[EdgeRecord] = []
        dynamics: list[EdgeRecord] = []

        for edge in stable_edges:
            if is_hub(edge.source_name) or is_hub(edge.target_name):
                identity.append(edge)
            else:
                dynamics.append(edge)

        # Sort both by recency within their group
        identity.sort(key=lambda e: e.recency_key or "", reverse=True)
        dynamics.sort(key=lambda e: e.recency_key or "", reverse=True)

        return identity, dynamics

    # ── Formatting (4-section structure) ─────────────────────────────────

    @staticmethod
    def _format_compilation(
        episode_lines: list[str],
        active_lines: list[str],
        node_lines: list[str],
        identity_edge_lines: list[str],
        dynamics_lines: list[str],
    ) -> str:
        sections = []

        if episode_lines:
            sections.append(
                "#### 1. SESSION HISTORY ####\n"
                "# (Chronological record of recent conversations — what was discussed and when)\n"
                + "\n".join(episode_lines)
            )

        if active_lines:
            sections.append(
                "#### 2. ACTIVE CONTEXT ####\n"
                "# (Recently updated facts and relationships — what's current and top of mind)\n"
                + "\n".join(active_lines)
            )

        if node_lines or identity_edge_lines:
            identity_parts = []
            if node_lines:
                identity_parts.extend(node_lines)
            if identity_edge_lines:
                identity_parts.extend(identity_edge_lines)
            sections.append(
                "#### 3. CORE IDENTITY & DEFINITIONS ####\n"
                "# (Stable facts about who this person is and their key relationships)\n"
                + "\n".join(identity_parts)
            )

        if dynamics_lines:
            sections.append(
                "#### 4. BEHAVIORAL PATTERNS & DYNAMICS ####\n"
                "# (Therapeutic patterns, coping mechanisms, and relational dynamics)\n"
                + "\n".join(dynamics_lines)
            )

        if not sections:
            return ""

        content = "\n\n".join(sections)

        total_chars = (
            sum(len(l) for l in episode_lines)
            + sum(len(l) for l in active_lines)
            + sum(len(l) for l in node_lines)
            + sum(len(l) for l in identity_edge_lines)
            + sum(len(l) for l in dynamics_lines)
        )
        est_tokens = total_chars // 4
        all_edges = len(active_lines) + len(identity_edge_lines) + len(dynamics_lines)
        stats = (
            f"\n\n### STATS ###\n"
            f"# Episodes: {len(episode_lines)} | Entities: {len(node_lines)} "
            f"| Relations: {all_edges} | Est. Tokens: ~{est_tokens}"
        )

        return content + stats
