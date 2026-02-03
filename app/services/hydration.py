"""
Hydration Service - Builds user knowledge compilation from Neo4j.

Uses direct Cypher queries (bypassing Graphiti) for performance.
Implements degree-based filtering to prioritize well-connected entities.
"""

from neo4j import AsyncDriver

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
        definitions = await self._fetch_entity_definitions(user_id)
        relationships = await self._fetch_relationships(user_id)

        if not definitions and not relationships:
            return ""

        return self._format_compilation(definitions, relationships)

    async def _fetch_entity_definitions(self, group_id: str) -> list[str]:
        """
        Fetch entity definitions ordered by connectivity.

        Only includes entities with at least min_degree connections,
        filtering out long-tail noise.
        """
        async with self.driver.session() as session:
            result = await session.run(
                FETCH_ENTITIES_QUERY,
                group_id=group_id,
                min_degree=self.min_degree,
            )
            records = await result.data()

        lines = []
        for record in records:
            name = record.get("name", "")
            summary = record.get("summary", "")
            if name and summary:
                lines.append(f"- **{name}**: {summary}")
        return lines

    async def _fetch_relationships(self, group_id: str) -> list[str]:
        """
        Fetch relationships with facts and temporal context.

        Returns formatted relationship lines including verb, fact, and timestamps.
        """
        async with self.driver.session() as session:
            result = await session.run(
                FETCH_RELATIONSHIPS_QUERY,
                group_id=group_id,
            )
            records = await result.data()

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
