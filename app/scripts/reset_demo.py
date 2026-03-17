#!/usr/bin/env python3
"""
Demo reset: deletes existing graph data for a group_id and re-inserts from
scripts/seed_data/demo_graph.json. No LLM calls — pure Neo4j batch insert.

    python scripts/reset_demo.py --group-id <clerk_user_id> [--dry-run]

Run this whenever you need to reset the demo user back to the seeded state.
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

SEED_FILE = Path(__file__).parent / "seed_data" / "demo_graph.json"


def apply_group_id(records: list[dict], group_id: str) -> list[dict]:
    return [{**r, "group_id": group_id} for r in records]


def parse_datetime_props(props: dict) -> dict:
    """
    Convert ISO datetime strings back to Python datetime objects so the Neo4j
    driver stores them as DateTime (not plain strings), preserving the temporal
    queries in the hydration service (e.g. WHERE r.invalid_at > datetime()).
    """
    datetime_fields = {"created_at", "updated_at", "valid_at", "invalid_at", "expired_at"}
    result = {}
    for k, v in props.items():
        if k in datetime_fields and isinstance(v, str) and v:
            try:
                result[k] = datetime.fromisoformat(v)
            except ValueError:
                result[k] = v
        else:
            result[k] = v
    return result


async def run(group_id: str, dry_run: bool) -> None:
    if not SEED_FILE.exists():
        logger.error(f"Seed file not found: {SEED_FILE}")
        logger.error("Run export_demo_graph.py first to generate it.")
        sys.exit(1)

    seed = json.loads(SEED_FILE.read_text())

    entity_nodes = [
        parse_datetime_props(r)
        for r in apply_group_id(seed["entity_nodes"], group_id)
    ]
    episodic_nodes = [
        parse_datetime_props(r)
        for r in apply_group_id(seed["episodic_nodes"], group_id)
    ]
    # Split routing info from edge properties for the Cypher query
    relates_to_edges = [
        {
            "source_uuid": e["source_uuid"],
            "target_uuid": e["target_uuid"],
            "props": parse_datetime_props({
                k: v for k, v in e.items() if k not in ("source_uuid", "target_uuid")
            } | {"group_id": group_id}),
        }
        for e in seed["relates_to_edges"]
    ]
    mentions_edges = seed["mentions_edges"]

    logger.info(
        f"{'[DRY RUN] ' if dry_run else ''}Reset demo for group_id={group_id!r}\n"
        f"  {len(entity_nodes)} Entity nodes\n"
        f"  {len(episodic_nodes)} Episodic nodes\n"
        f"  {len(relates_to_edges)} RELATES_TO edges\n"
        f"  {len(mentions_edges)} MENTIONS edges"
    )

    if dry_run:
        logger.info("Dry run complete — no changes made.")
        return

    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    try:
        await driver.verify_connectivity()

        async with driver.session() as neo4j_session:
            # 1. Delete existing data for this group_id
            result = await neo4j_session.run(
                "MATCH (n {group_id: $gid}) DETACH DELETE n",
                gid=group_id,
            )
            summary = await result.consume()
            logger.info(f"Deleted {summary.counters.nodes_deleted} existing nodes")

            # 2. Insert Entity nodes
            await neo4j_session.run(
                "UNWIND $nodes AS n MERGE (e:Entity {uuid: n.uuid}) SET e += n",
                nodes=entity_nodes,
            )
            logger.info(f"Inserted {len(entity_nodes)} Entity nodes")

            # 3. Insert Episodic nodes
            await neo4j_session.run(
                "UNWIND $nodes AS n MERGE (e:Episodic {uuid: n.uuid}) SET e += n",
                nodes=episodic_nodes,
            )
            logger.info(f"Inserted {len(episodic_nodes)} Episodic nodes")

            # 4. Insert RELATES_TO edges
            await neo4j_session.run(
                """
                UNWIND $edges AS e
                MATCH (s:Entity {uuid: e.source_uuid})
                MATCH (t:Entity {uuid: e.target_uuid})
                MERGE (s)-[r:RELATES_TO {uuid: e.props.uuid}]->(t)
                SET r += e.props
                """,
                edges=relates_to_edges,
            )
            logger.info(f"Inserted {len(relates_to_edges)} RELATES_TO edges")

            # 5. Insert MENTIONS edges
            await neo4j_session.run(
                """
                UNWIND $edges AS e
                MATCH (ep:Episodic {uuid: e.source_uuid})
                MATCH (en:Entity {uuid: e.target_uuid})
                MERGE (ep)-[:MENTIONS]->(en)
                """,
                edges=mentions_edges,
            )
            logger.info(f"Inserted {len(mentions_edges)} MENTIONS edges")

        logger.info(f"\nDemo reset complete for group_id={group_id!r}")

    finally:
        await driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset demo user graph from seed snapshot")
    parser.add_argument("--group-id", required=True, help="Clerk user ID of the demo account")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing to Neo4j",
    )
    args = parser.parse_args()
    asyncio.run(run(args.group_id, args.dry_run))


if __name__ == "__main__":
    main()
