"""
Hydration Service - Builds user knowledge compilation from Neo4j.

Uses direct Cypher queries (bypassing Graphiti) for performance.
Implements degree-based filtering to prioritize well-connected entities.
"""

import time

from neo4j import AsyncDriver
from opentelemetry import trace

from app.core.observability import (
    anonymize_id,
    classify_error,
    mark_span_error,
    mark_span_success,
    set_span_attributes,
)

# Default minimum connections for entity inclusion (filters long-tail noise)
DEFAULT_MIN_DEGREE = 2


# Cypher: Fetch entity definitions ordered by connectivity (most connected first)
FETCH_ENTITIES_QUERY = """
MATCH (n:Entity)-[r:RELATES_TO]-(other:Entity)
WHERE n <> other
  AND n.group_id = $group_id
  AND n.summary IS NOT NULL
  AND n.summary <> ""
WITH n, count(r) AS degree
WHERE degree >= $min_degree
RETURN n.name AS name, n.summary AS summary, degree
ORDER BY degree DESC
"""

# Cypher: Fetch relationships with temporal context
FETCH_RELATIONSHIPS_QUERY = """
MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
WHERE r.group_id = $group_id
  AND (r.invalid_at IS NULL OR r.invalid_at > datetime())
  AND NOT 'Episode' IN labels(source)
  AND NOT 'Episode' IN labels(target)
RETURN 
    source.name AS source_name,
    r.name AS relation_name,
    target.name AS target_name,
    r.fact AS fact,
    r.valid_at AS valid_at,
    r.invalid_at AS invalid_at
ORDER BY r.valid_at DESC
"""


class HydrationService:
    """Service for building user knowledge compilation from the knowledge graph."""

    def __init__(self, driver: AsyncDriver, min_degree: int = DEFAULT_MIN_DEGREE):
        self.driver = driver
        self.min_degree = min_degree

    async def build_user_knowledge(self, user_id: str) -> str:
        """
        Build a user knowledge compilation from the knowledge graph.

        Args:
            user_id: The user's ID (used as group_id in the graph).

        Returns:
            A structured text compilation of the user's knowledge profile.
        """
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "hydration.build_user_knowledge",
            attributes={
                "hydrate.user_id": anonymize_id(user_id),
                "hydrate.min_degree": self.min_degree,
            },
        ) as span:
            start = time.monotonic()
            try:
                definitions = await self._fetch_entity_definitions(user_id)
                relationships = await self._fetch_relationships(user_id)

                if not definitions and not relationships:
                    set_span_attributes(
                        span,
                        {
                            "hydrate.definitions_count": 0,
                            "hydrate.relationships_count": 0,
                            "hydrate.compilation_size_chars": 0,
                            "duration_ms": round((time.monotonic() - start) * 1000, 2),
                        },
                    )
                    mark_span_success(span)
                    return ""

                compilation = self._format_compilation(definitions, relationships)
                set_span_attributes(
                    span,
                    {
                        "hydrate.definitions_count": len(definitions),
                        "hydrate.relationships_count": len(relationships),
                        "hydrate.compilation_size_chars": len(compilation),
                        "duration_ms": round((time.monotonic() - start) * 1000, 2),
                    },
                )
                mark_span_success(span)
                return compilation
            except Exception as e:
                category, code = classify_error(e)
                mark_span_error(span, e, category=category, code=code)
                raise

    async def _fetch_entity_definitions(self, group_id: str) -> list[str]:
        """
        Fetch entity definitions ordered by connectivity.

        Only includes entities with at least min_degree connections,
        filtering out long-tail noise.
        """
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "db.cypher.fetch_entities",
            attributes={
                "db.system": "neo4j",
                "db.query_type": "fetch_entities",
                "hydrate.group_id": anonymize_id(group_id),
            },
        ) as span:
            start = time.monotonic()
            try:
                async with self.driver.session() as session:
                    result = await session.run(
                        FETCH_ENTITIES_QUERY,
                        group_id=group_id,
                        min_degree=self.min_degree,
                    )
                    records = await result.data()
            except Exception as e:
                category, code = classify_error(e)
                mark_span_error(span, e, category=category, code=code)
                raise

            lines = []
            for record in records:
                name = record.get("name", "")
                summary = record.get("summary", "")
                if name and summary:
                    lines.append(f"- **{name}**: {summary}")

            set_span_attributes(
                span,
                {
                    "db.records_returned": len(records),
                    "hydrate.definitions_count": len(lines),
                    "db.query_duration_ms": round((time.monotonic() - start) * 1000, 2),
                },
            )
            mark_span_success(span)
            return lines

    async def _fetch_relationships(self, group_id: str) -> list[str]:
        """
        Fetch relationships with facts and temporal context.

        Returns formatted relationship lines including verb, fact, and timestamps.
        """
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            "db.cypher.fetch_relationships",
            attributes={
                "db.system": "neo4j",
                "db.query_type": "fetch_relationships",
                "hydrate.group_id": anonymize_id(group_id),
            },
        ) as span:
            start = time.monotonic()
            try:
                async with self.driver.session() as session:
                    result = await session.run(
                        FETCH_RELATIONSHIPS_QUERY,
                        group_id=group_id,
                    )
                    records = await result.data()
            except Exception as e:
                category, code = classify_error(e)
                mark_span_error(span, e, category=category, code=code)
                raise

            lines = []
            for record in records:
                source = record.get("source_name", "")
                target = record.get("target_name", "")
                verb = self._format_relation_name(record.get("relation_name"))
                fact = record.get("fact")
                valid_at = record.get("valid_at")
                invalid_at = record.get("invalid_at")

                if not source or not target:
                    continue

                line = f"- {source} {verb} {target}"

                if fact:
                    line += f': "{fact}"'

                # Add temporal context if available
                temporal_parts = []
                if valid_at:
                    temporal_parts.append(f"valid_at: {valid_at}")
                if invalid_at:
                    temporal_parts.append(f"invalid_at: {invalid_at}")
                if temporal_parts:
                    line += f" [{', '.join(temporal_parts)}]"

                lines.append(line)

            set_span_attributes(
                span,
                {
                    "db.records_returned": len(records),
                    "hydrate.relationships_count": len(lines),
                    "db.query_duration_ms": round((time.monotonic() - start) * 1000, 2),
                },
            )
            mark_span_success(span)
            return lines

    @staticmethod
    def _format_relation_name(name: str | None) -> str:
        """Convert relation name to readable verb (e.g., WORKS_WITH -> works with)."""
        if not name:
            return "relates to"
        return name.replace("_", " ").lower()

    @staticmethod
    def _format_compilation(definitions: list[str], relationships: list[str]) -> str:
        """
        Format definitions and relationships into a structured compilation.

        Output format optimized for LLM consumption with clear sections.
        """
        sections = []

        if definitions:
            sections.append(
                "#### 1. CONCEPTUAL DEFINITIONS & IDENTITY ####\n"
                "# (Understanding what these concepts mean specifically for this user)\n"
                + "\n".join(definitions)
            )

        if relationships:
            sections.append(
                "#### 2. RELATIONAL DYNAMICS & CAUSALITY ####\n"
                "# (How these concepts interact and evolve over time)\n"
                + "\n".join(relationships)
            )

        if not sections:
            return ""

        content = "\n\n".join(sections)

        # Add stats footer
        total_chars = sum(len(d) for d in definitions) + sum(len(r) for r in relationships)
        est_tokens = total_chars // 4
        stats = f"\n\n### STATS ###\n# Definitions: {len(definitions)} | Relations: {len(relationships)} | Est. Tokens: ~{est_tokens}"

        return content + stats
