#!/usr/bin/env python3
"""
Rails-style console: interactive Python shell with Graphiti and project services
already set up. Run from project root:

    python scripts/console.py
    # or: uv run python scripts/console.py

Then type code as you would in a REPL. Available in the shell:

  graphiti       - Graphiti client (search, add_episode, _search, etc.)
  neo4j_driver   - Neo4j AsyncDriver
  graph_service  - GraphService (get_graph, correct_memory)
  hydration_service - HydrationService (get_compilation)
  ingestion_service - IngestionService
  generation_service - GenerationService
  settings       - App settings (get_settings())

Examples:

  # Graphiti hybrid search (use await in IPython)
  edges = await graphiti.search("what are my preferences?", group_id="user-123")
  [e.fact for e in edges]

  # Low-level search with a recipe
  from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
  out = await graphiti._search("visa", group_id="user-123", config=EDGE_HYBRID_SEARCH_RRF)

  # Project services (async)
  g = await graph_service.get_graph("user-123")
  comp = await hydration_service.get_compilation("user-123", max_entities=20)

Requires IPython (in requirements.txt) so you can use top-level await.
"""

import asyncio
import sys
from pathlib import Path

# Require IPython so top-level await works. Check before loading the rest.
try:
    from IPython import embed
except ImportError:
    print("IPython is required for the console (so 'await' works in the REPL).")
    print("Install it with:  pip install ipython")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import AsyncGraphDatabase

from app.core.config import get_settings
from app.services.generation import GenerationService
from app.services.graph import GraphService
from app.services.hydration import HydrationService
from app.services.ingestion import IngestionService


def _make_graphiti():
    from graphiti_core import Graphiti
    from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
    from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

    s = get_settings()
    return Graphiti(
        s.neo4j_uri,
        s.neo4j_user,
        s.neo4j_password,
        llm_client=GeminiClient(
            config=LLMConfig(api_key=s.google_api_key, model=s.graphiti_model)
        ),
        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=s.google_api_key,
                embedding_model="gemini-embedding-001",
            )
        ),
        cross_encoder=GeminiRerankerClient(
            config=LLMConfig(api_key=s.google_api_key, model=s.graphiti_model)
        ),
        max_coroutines=s.semaphore_limit,
    )


async def _setup():
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    await driver.verify_connectivity()

    graphiti = _make_graphiti()
    await graphiti.build_indices_and_constraints()

    hydration_service = HydrationService(driver)
    ingestion_service = IngestionService(graphiti, hydration_service, settings.graphiti_model)
    generation_service = GenerationService(settings.google_api_key)
    graph_service = GraphService(driver, graphiti)

    loop = asyncio.get_running_loop()

    def run(coro):
        """Run an async coroutine (e.g. run(graphiti.search('q'))). Uses the same loop as the console."""
        return loop.run_until_complete(coro)

    return {
        "graphiti": graphiti,
        "neo4j_driver": driver,
        "graph_service": graph_service,
        "hydration_service": hydration_service,
        "ingestion_service": ingestion_service,
        "generation_service": generation_service,
        "settings": settings,
        "asyncio": asyncio,
        "run": run,
        "loop": loop,
    }


def _run_console(ns: dict) -> None:
    from traitlets.config import Config
    from IPython.terminal.embed import InteractiveShellEmbed

    loop = ns["loop"]
    # Force IPython to run awaited coroutines on OUR loop (same as Neo4j/Graphiti).
    # Otherwise IPython uses get_event_loop() which can be a different loop → "attached to a different loop".
    def loop_runner(coro):
        return loop.run_until_complete(coro)

    c = Config()
    c.InteractiveShell.autoawait = True
    c.InteractiveShell.loop_runner = loop_runner
    c.TerminalInteractiveShell.autoawait = True
    shell = InteractiveShellEmbed(
        config=c,
        user_ns=ns,
        colors="neutral",
        banner1="Synapse Cortex console (Graphiti + services loaded). Use exit() or Ctrl-D to leave.",
    )
    shell.autoawait = True
    shell.loop_runner = loop_runner
    shell()


def main():
    # Use one event loop for setup AND for IPython so await in the REPL uses the same
    # loop the Neo4j driver and Graphiti were created on (avoids "attached to a different loop").
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        print("Loading Neo4j + Graphiti + services...")
        ns = loop.run_until_complete(_setup())
        print("Ready. Dropping into shell.")
        _run_console(ns)
    finally:
        print("Closing Graphiti and Neo4j...")
        loop.run_until_complete(ns["graphiti"].close())
        loop.run_until_complete(ns["neo4j_driver"].close())
        loop.close()
        print("Done.")


if __name__ == "__main__":
    main()
