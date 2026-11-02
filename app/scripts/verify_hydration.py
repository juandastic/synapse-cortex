"""Explicit local integration check with real Neo4j and configured providers.

Run: python -m app.scripts.verify_hydration --env-file /path/to/.env
Only loopback Neo4j is allowed. A synthetic user is created and removed.
--user USER instead compiles existing local data without modifying its graph.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from fastapi import FastAPI
from graphiti_core.llm_client.config import LLMConfig
from neo4j import AsyncGraphDatabase

from app.api.routes import router
from app.core.config import HYDRATION_MODEL, Settings, create_genai_client
from app.core.graphiti_client import create_graphiti
from app.core.graphiti_llm import OpenRouterGraphitiClient
from app.schemas.models import IngestRequest
from app.services.cache_manager import CacheManager
from app.services.generation import GenerationService
from app.services.hydration import HydrationService
from app.services.hydration_compiler import HydrationCompiler
from app.services.hydration_store import DELETE_HYDRATION_QUERY, HydrationStore
from app.services.ingestion import IngestionService
from app.services.job_store import create_job, get_job, remove_job
from app.services.openrouter_generation import OpenRouterGenerationService
from app.services.persisted_hydration import PersistedHydration


async def main(args):
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    logging.getLogger("app.services.persisted_hydration").setLevel(logging.INFO)
    settings = Settings(_env_file=args.env_file)
    if urlsplit(settings.neo4j_uri).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Integration verification only permits a local Neo4j")
    if not settings.open_router_api_key:
        raise ValueError("OPEN_ROUTER_API_KEY is required for hydration compilation")
    # Router authentication uses the same already-loaded settings, without printing secrets.
    import app.core.security as security

    security.get_settings = lambda: settings
    user = args.user or f"hydration-integration-{uuid4()}"
    client = create_genai_client(settings)
    graphiti = create_graphiti(settings, client)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    store = HydrationStore(driver)
    legacy = HydrationService(driver)
    hydration_llm = OpenRouterGraphitiClient(
        LLMConfig(
            api_key=settings.open_router_api_key,
            model=HYDRATION_MODEL,
            small_model=HYDRATION_MODEL,
            max_tokens=16384,
        ),
        reasoning_effort="low",
    )
    compiler = HydrationCompiler(hydration_llm)
    worker = PersistedHydration(store, compiler, legacy)
    legacy.persisted = worker
    await store.initialize()
    app = FastAPI()
    app.include_router(router)
    app.state.hydration_service = legacy
    app.state.ingestion_service = IngestionService(
        graphiti, settings.graphiti_model, worker.changed
    )
    app.state.cache_manager = CacheManager(client, settings.chat_model, grounding_enabled=False)
    worker.cache_manager = app.state.cache_manager
    app.state.graphiti = graphiti
    app.state.generation_service = GenerationService(client, grounding_enabled=False)
    app.state.openrouter_generation_service = OpenRouterGenerationService(
        settings.open_router_api_key
    )
    headers = {"X-API-SECRET": settings.synapse_api_secret}
    started = time.monotonic()
    report = {
        "user": user,
        "real_neo4j": True,
        "real_llm": True,
        "graph_provider": settings.graphiti_provider,
        "compiler_provider": "openrouter",
        "compiler_model": HYDRATION_MODEL,
        "chat_model": settings.chat_model,
    }
    job_id = f"integration-{uuid4()}"
    try:
        await graphiti.build_indices_and_constraints()
        if not args.user:
            now = int(time.time() * 1000)
            request = IngestRequest(
                jobId=job_id,
                userId=user,
                sessionId=job_id,
                messages=[
                    dict(
                        role="user",
                        timestamp=now,
                        content=(
                            "Me llamo Elena. Vivo en Bogotá y trabajo como diseñadora. "
                            "Martina es mi sobrina, no mi hija. Mi prima Julia estudia ingeniería. "
                            "Quiero estudiar fotografía el próximo año. Prefiero explicaciones concretas "
                            "con ejemplos. Mi perro se llama Bruno y tenemos una cita veterinaria el viernes."
                        ),
                    )
                ],
                metadata=dict(sessionStartedAt=now, sessionEndedAt=now, messageCount=1),
            )
            create_job(job_id, user, job_id)
            await asyncio.wait_for(
                app.state.ingestion_service._process_background(job_id, request), timeout=600
            )
            assert get_job(job_id).status == "completed", get_job(job_id).error
            report["ingestion_completed"] = True

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as http:
            cold = await http.post(
                "/hydrate", headers=headers, json={"userId": user, "version": "v2"}
            )
            cold.raise_for_status()
            report["cold_fallback_ok"] = cold.json()["success"]
            # Manual drain proves durable state, without relying on an in-memory task.
            current = await store.read(user)
            if current is None:
                await worker.changed(user, invalidate=True)
            deadline = time.monotonic() + 2400
            slices = 0
            while current is None and time.monotonic() < deadline:
                if not await worker.run_one(user=user):
                    state = await store.rows(
                        "MATCH (n:SynapseUserCore {user_id:$user}) RETURN "
                        "n.hydration_failed_generation=n.hydration_generation AS failed",
                        user=user,
                    )
                    assert not state or not state[0]["failed"], "Build exhausted retries"
                    await asyncio.sleep(1)
                else:
                    slices += 1
                current = await store.read(user)
            report["build_slices"] = slices
            current = await store.read(user)
            assert current is not None, "No compilation published"
            hydrated = await http.post(
                "/hydrate", headers=headers, json={"userId": user, "version": "v2"}
            )
            hydrated.raise_for_status()
            body = hydrated.json()
            revision = body["compilationMetadata"]["compilation_revision"]
            assert revision
            again = await http.post(
                "/hydrate", headers=headers, json={"userId": user, "version": "v2"}
            )
            assert again.json()["compilationMetadata"]["compilation_revision"] == revision
            assert again.json()["userKnowledgeCompilation"] == body["userKnowledgeCompilation"]
            # A newly constructed service reads exactly the same stored revision.
            restarted = HydrationStore(driver)
            assert (await restarted.read(user)).compilation_text == current.compilation_text
            report.update(
                revision=revision,
                chars=len(current.compilation_text),
                included_edges=len(current.metadata.included_edge_ids),
                source_episodes=len(current.metadata.included_episode_ids),
                repeat_reads_same_revision=True,
                restart_read_ok=True,
            )
            if not args.user:
                # Only synthetic fixture data is included in assertion diagnostics.
                _, fixture_edges = await store.snapshot(user)
                fixture_nodes = await store.rows(
                    "MATCH (n:Entity {group_id:$user}) RETURN n.summary AS summary",
                    user=user,
                )
                diagnostic = {
                    "compiled": current.compilation_text,
                    "active_facts": [edge["fact"] for edge in fixture_edges],
                    "entity_summaries": [node["summary"] or "" for node in fixture_nodes],
                }
                # The core is deliberately partial. Require the fixture's facts
                # in the graph, and the complete answer below through core + RAG,
                # rather than requiring one model's particular core selection.
                graph_text = "\n".join(diagnostic["active_facts"] + diagnostic["entity_summaries"])
                assert any(word in graph_text.lower() for word in ("sobrina", "niece")), diagnostic
                assert "Julia" in graph_text, diagnostic
                assert any(word in graph_text.lower() for word in ("prima", "cousin")), diagnostic
                assert any(word in graph_text.lower() for word in ("ingenier", "engineer")), (
                    diagnostic
                )
                chat = await http.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={
                        "user_id": user,
                        "provider": "vertex",
                        "model": settings.chat_model,
                        "system_instruction": "Responde breve en español. Usa la memoria, no inventes.",
                        "compilation": body["userKnowledgeCompilation"],
                        "compilationMetadata": body["compilationMetadata"],
                        "cache_name": body.get("cacheName"),
                        "messages": [
                            {
                                "role": "user",
                                "content": "¿Quiénes son Martina y Julia para mí, y qué estudia Julia?",
                            }
                        ],
                    },
                )
                chat.raise_for_status()
                assert "[DONE]" in chat.text and '"error"' not in chat.text, chat.text[:300]
                report["chat_stream_completed"] = True
                content = "".join(
                    (json.loads(line[6:])["choices"][0]["delta"].get("content") or "")
                    for line in chat.text.splitlines()
                    if line.startswith("data: {") and json.loads(line[6:]).get("choices")
                )
                report["chat_answer"] = content
                assert all(word in content.lower() for word in ("sobrina", "prima", "ingenier")), (
                    content
                )
            if args.output:
                Path(args.output).write_text(body["userKnowledgeCompilation"])
            report["seconds"] = round(time.monotonic() - started, 2)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        remove_job(job_id)
        if not args.user:
            await store.rows("MATCH (n {group_id:$user}) DETACH DELETE n", user=user)
            await store.rows(DELETE_HYDRATION_QUERY, user=user)
        await app.state.openrouter_generation_service.close()
        await hydration_llm.close()
        await graphiti.close()
        if hasattr(graphiti.llm_client, "close"):
            await graphiti.llm_client.close()
        await driver.close()
        await client.aio.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--user")
    parser.add_argument("--output")
    asyncio.run(main(parser.parse_args()))
