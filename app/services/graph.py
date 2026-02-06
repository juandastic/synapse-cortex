"""
Graph Service - Knowledge graph visualization and memory correction.

Provides:
1. Graph retrieval for react-force-graph visualization (via direct Cypher queries)
2. Memory correction via Graphiti's add_episode (preserves embeddings and temporal integrity)
"""

import logging
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from neo4j import AsyncDriver

from app.schemas.models import GraphLink, GraphNode, GraphResponse

logger = logging.getLogger(__name__)


# Cypher: Fetch entity nodes with connection count for visual sizing
FETCH_GRAPH_NODES_QUERY = """
MATCH (n:Entity)-[r:RELATES_TO]-(other:Entity)
WHERE n <> other
  AND n.group_id = $group_id
  AND n.summary IS NOT NULL
  AND n.summary <> ""
WITH n, count(r) AS degree
RETURN n.uuid AS id, n.name AS name, degree AS val, n.summary AS summary
ORDER BY degree DESC
"""

# Cypher: Fetch valid (non-expired) relationships, excluding Episode nodes
FETCH_GRAPH_LINKS_QUERY = """
MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
WHERE r.group_id = $group_id
  AND (r.invalid_at IS NULL OR r.invalid_at > datetime())
  AND NOT 'Episode' IN labels(source)
  AND NOT 'Episode' IN labels(target)
RETURN source.uuid AS source, target.uuid AS target,
       r.name AS label, r.fact AS fact
"""


class GraphService:
    """Service for knowledge graph visualization and memory correction."""

    def __init__(self, driver: AsyncDriver, graphiti: Graphiti):
        self.driver = driver
        self.graphiti = graphiti

    async def get_graph(self, group_id: str) -> GraphResponse:
        """
        Retrieve the knowledge graph for a user in react-force-graph format.

        Args:
            group_id: The user/group ID to fetch the graph for.

        Returns:
            GraphResponse with nodes and links ready for react-force-graph-2d.
        """
        nodes = await self._fetch_nodes(group_id)
        links = await self._fetch_links(group_id)

        return GraphResponse(nodes=nodes, links=links)

    async def correct_memory(self, group_id: str, correction_text: str) -> None:
        """
        Apply a memory correction via Graphiti's add_episode.

        This preserves embeddings, temporal integrity, and relationship consistency.
        Graphiti will automatically invalidate outdated edges and create new ones.

        Args:
            group_id: The user/group ID whose memory to correct.
            correction_text: Natural language correction from the user.
        """
        logger.info(f"Applying memory correction for group {group_id}")

        await self.graphiti.add_episode(
            name="user_memory_correction",
            episode_body=correction_text,
            source=EpisodeType.text,
            source_description="User-initiated memory correction via Memory Explorer",
            group_id=group_id,
            reference_time=datetime.now(),
        )

        logger.info(f"Memory correction applied successfully for group {group_id}")

    async def _fetch_nodes(self, group_id: str) -> list[GraphNode]:
        """Fetch entity nodes with their connection count for visual sizing."""
        async with self.driver.session() as session:
            result = await session.run(
                FETCH_GRAPH_NODES_QUERY,
                group_id=group_id,
            )
            records = await result.data()

        nodes = []
        for record in records:
            node_id = record.get("id", "")
            name = record.get("name", "")
            if node_id and name:
                nodes.append(
                    GraphNode(
                        id=node_id,
                        name=name,
                        val=record.get("val", 1),
                        summary=record.get("summary", ""),
                    )
                )
        return nodes

    async def _fetch_links(self, group_id: str) -> list[GraphLink]:
        """Fetch valid relationships between entities."""
        async with self.driver.session() as session:
            result = await session.run(
                FETCH_GRAPH_LINKS_QUERY,
                group_id=group_id,
            )
            records = await result.data()

        links = []
        for record in records:
            source = record.get("source", "")
            target = record.get("target", "")
            if source and target:
                links.append(
                    GraphLink(
                        source=source,
                        target=target,
                        label=record.get("label", "RELATES_TO"),
                        fact=record.get("fact"),
                    )
                )
        return links
