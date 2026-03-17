#!/usr/bin/env python3
"""
Demo ingest: reads scripts/seed_demo.json and runs graphiti.add_episode()
for each session. This is the expensive step (costs LLM tokens) — run once.

    python scripts/ingest_demo.py [--group-id CUSTOM_GROUP_ID]

Default group_id: demo_seed_YYYYMMDD (today's date).
After running, use export_demo_graph.py to snapshot the resulting graph.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.nodes import EpisodeType

from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED_FILE = Path(__file__).parent / "seed_demo.json"


def make_graphiti(settings) -> Graphiti:
    return Graphiti(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        llm_client=GeminiClient(
            config=LLMConfig(api_key=settings.google_api_key, model=settings.graphiti_model)
        ),
        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=settings.google_api_key,
                embedding_model="gemini-embedding-001",
            )
        ),
        cross_encoder=GeminiRerankerClient(
            config=LLMConfig(api_key=settings.google_api_key, model=settings.graphiti_model)
        ),
        max_coroutines=settings.semaphore_limit,
    )


def format_messages(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{label}: {msg['content']}")
    return "\n\n".join(lines)


async def run(group_id: str) -> None:
    settings = get_settings()

    logger.info(f"Loading seed data from {SEED_FILE}")
    seed = json.loads(SEED_FILE.read_text())

    sessions = [
        session
        for thread in seed["threads"]
        for session in thread["sessions"]
    ]
    logger.info(f"Found {len(sessions)} sessions to ingest for group_id={group_id!r}")

    graphiti = make_graphiti(settings)
    await graphiti.build_indices_and_constraints()

    total_nodes = 0
    total_edges = 0

    try:
        for i, session in enumerate(sessions, 1):
            session_id = session["localId"]
            episode_name = f"session_{session_id}"
            messages = session["messages"]
            ended_at = session["endedAt"]
            reference_time = datetime.fromtimestamp(ended_at / 1000, tz=timezone.utc)
            episode_body = format_messages(messages)

            logger.info(f"[{i}/{len(sessions)}] Ingesting {episode_name} ({len(messages)} messages)...")
            t0 = time.monotonic()

            result = await graphiti.add_episode(
                name=episode_name,
                episode_body=episode_body,
                source=EpisodeType.message,
                source_description="Chat conversation from Synapse demo seed",
                group_id=group_id,
                reference_time=reference_time,
            )

            elapsed_ms = (time.monotonic() - t0) * 1000
            total_nodes += len(result.nodes)
            total_edges += len(result.edges)
            logger.info(f"  → {len(result.nodes)} nodes, {len(result.edges)} edges ({elapsed_ms:.0f}ms)")

    finally:
        await graphiti.close()

    logger.info(
        f"\nDone. {total_nodes} nodes, {total_edges} edges ingested for group_id={group_id!r}"
    )
    logger.info(f"Next: python scripts/export_demo_graph.py --group-id {group_id!r}")


def main() -> None:
    today = datetime.now().strftime("%Y%m%d")
    default_group = f"demo_seed_{today}"

    parser = argparse.ArgumentParser(description="Ingest demo seed data into Neo4j via Graphiti")
    parser.add_argument(
        "--group-id",
        default=default_group,
        help=f"group_id to use (default: {default_group})",
    )
    args = parser.parse_args()
    asyncio.run(run(args.group_id))


if __name__ == "__main__":
    main()
