#!/usr/bin/env python3
"""
Demo export: reads Entity/Episodic nodes and RELATES_TO/MENTIONS edges from
Neo4j for a given group_id, replaces group_id with DEMO_PLACEHOLDER, and
writes the snapshot to scripts/seed_data/demo_graph.json.

    python scripts/export_demo_graph.py --group-id demo_seed_YYYYMMDD [--delete-after]

--delete-after removes the temporary group_id from Neo4j after export.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import AsyncGraphDatabase

from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PLACEHOLDER = "DEMO_PLACEHOLDER"
OUTPUT_FILE = Path(__file__).parent / "seed_data" / "demo_graph.json"


def neo4j_to_python(val):
    """Recursively convert Neo4j types to JSON-serializable Python values."""
    # Import lazily to avoid hard dependency when neo4j is not installed
    try:
        from neo4j.time import DateTime, Date, Duration, Time
        if isinstance(val, (DateTime, Date, Time)):
            return val.iso_format()
        if isinstance(val, Duration):
            return str(val)
    except ImportError:
        pass

    if isinstance(val, list):
        return [neo4j_to_python(v) for v in val]
    if isinstance(val, dict):
        return {k: neo4j_to_python(v) for k, v in val.items()}
    return val


def serialize_props(props: dict) -> dict:
    """Serialize properties, replacing the real group_id with PLACEHOLDER."""
    result = {}
    for k, v in props.items():
        v = neo4j_to_python(v)
        if k == "group_id":
            v = PLACEHOLDER
        result[k] = v
    return result


async def run(group_id: str, delete_after: bool) -> None:
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    try:
        await driver.verify_connectivity()
        logger.info(f"Connected to Neo4j, exporting group_id={group_id!r}")

        async with driver.session() as neo4j_session:
            # Entity nodes
            result = await neo4j_session.run(
                "MATCH (n:Entity {group_id: $gid}) RETURN properties(n) AS props",
                gid=group_id,
            )
            entity_nodes = [serialize_props(r["props"]) async for r in result]
            logger.info(f"  Entity nodes:    {len(entity_nodes)}")

            # Episodic nodes
            result = await neo4j_session.run(
                "MATCH (n:Episodic {group_id: $gid}) RETURN properties(n) AS props",
                gid=group_id,
            )
            episodic_nodes = [serialize_props(r["props"]) async for r in result]
            logger.info(f"  Episodic nodes:  {len(episodic_nodes)}")

            # RELATES_TO edges
            result = await neo4j_session.run(
                """
                MATCH (s:Entity {group_id: $gid})-[r:RELATES_TO]->(t:Entity {group_id: $gid})
                RETURN properties(r) AS props, s.uuid AS source_uuid, t.uuid AS target_uuid
                """,
                gid=group_id,
            )
            relates_to_edges = []
            async for r in result:
                edge = serialize_props(r["props"])
                edge["source_uuid"] = r["source_uuid"]
                edge["target_uuid"] = r["target_uuid"]
                relates_to_edges.append(edge)
            logger.info(f"  RELATES_TO edges: {len(relates_to_edges)}")

            # MENTIONS edges (no properties to serialize, just connectivity)
            result = await neo4j_session.run(
                """
                MATCH (ep:Episodic {group_id: $gid})-[:MENTIONS]->(en:Entity {group_id: $gid})
                RETURN ep.uuid AS source_uuid, en.uuid AS target_uuid
                """,
                gid=group_id,
            )
            mentions_edges = [
                {"source_uuid": r["source_uuid"], "target_uuid": r["target_uuid"]}
                async for r in result
            ]
            logger.info(f"  MENTIONS edges:  {len(mentions_edges)}")

        output = {
            "schema_version": "1",
            "exported_at": datetime.now(tz=timezone.utc).isoformat(),
            "source_group_id": PLACEHOLDER,
            "entity_nodes": entity_nodes,
            "episodic_nodes": episodic_nodes,
            "relates_to_edges": relates_to_edges,
            "mentions_edges": mentions_edges,
        }

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        logger.info(f"\nExported to {OUTPUT_FILE}")

        if delete_after:
            async with driver.session() as neo4j_session:
                await neo4j_session.run(
                    "MATCH (n {group_id: $gid}) DETACH DELETE n",
                    gid=group_id,
                )
            logger.info(f"Deleted temporary group_id={group_id!r} from Neo4j")

        logger.info("Next: python scripts/reset_demo.py --group-id <clerk_user_id>")

    finally:
        await driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export demo graph snapshot from Neo4j")
    parser.add_argument("--group-id", required=True, help="group_id to export from Neo4j")
    parser.add_argument(
        "--delete-after",
        action="store_true",
        help="Delete the temporary group_id from Neo4j after export",
    )
    args = parser.parse_args()
    asyncio.run(run(args.group_id, args.delete_after))


if __name__ == "__main__":
    main()
