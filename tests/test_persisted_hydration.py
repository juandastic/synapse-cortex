from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import router
from app.core import security
from app.schemas.models import ChatCompletionRequest
from app.services.hydration import HydrationService
from app.services.hydration_compiler import (
    HydrationCompiler,
    Source,
    sources_from_episodes,
)
from app.services.hydration_result import CompilationMetadata, GraphStats, HydrationResult
from app.services.hydration_store import HydrationStore
from app.services.persisted_hydration import PersistedHydration


def claim(quote="Mi prima Julia es quien estudia ingeniería en Bogotá.", **extra):
    return dict(
        id="claim",
        source_id="ep:0:0",
        episode_id="ep",
        date="2026-10-30",
        attribution="usuario",
        quote=quote,
        section="Personas y vínculos",
        kind="relato",
        **extra,
    )


def published(text="Current personal memory"):
    return HydrationResult(
        text, CompilationMetadata(True, 10, compilation_revision="revision"), GraphStats(1, 1)
    )


def test_source_parser_keeps_all_user_text_and_imports_without_assistant_assertions():
    text = "Mi prima estudia ingeniería. " * 1200
    sources = sources_from_episodes(
        [
            dict(
                uuid="ep",
                valid_at="2026-10-30",
                source="message",
                content=f"User: {text}\n\nAssistant: You are an engineer.\n\nUser: Soy diseñadora.",
            ),
            dict(uuid="doc", valid_at="2026-10-01", source="text", content="Perfil aportado " * 10),
        ]
    )
    assert all("You are an engineer" not in s.text for s in sources)
    assert any("Soy diseñadora" in s.text for s in sources)
    assert any(s.attribution == "documento aportado" for s in sources)
    assert any(s.id == "ep:0:28000" for s in sources)
    assert sources[0].hash != Source("ep:0:0", "ep", "2026-10-30", "usuario", "changed").hash


def test_explicit_exclusions_and_technical_tests_are_not_compiled():
    sources = sources_from_episodes(
        [
            dict(
                uuid="ep",
                source="message",
                valid_at="2026-11-01",
                content="User: No guardes este mensaje como dato personal: vivo en París.\n\n"
                "User: Prueba técnica de integración: Qwen listo.\n\n"
                "User: Prefiero explicaciones concretas con ejemplos.",
            )
        ]
    )
    assert len(sources) == 1
    assert sources[0].text == "Prefiero explicaciones concretas con ejemplos."


async def test_current_text_still_updates_coverage_metadata():
    service = HydrationService(None)
    service.persisted = SimpleNamespace(
        store=SimpleNamespace(chat_read=AsyncMock(return_value=("updated", published())))
    )
    request = ChatCompletionRequest(
        user_id="owner",
        messages=[],
        compilation="Current personal memory",
        compilationMetadata=asdict(published().metadata),
        cache_name="cachedContents/same-text",
    )
    assert not await service.refresh_chat_compilation(request)
    assert request.compilationMetadata.compilation_revision == "revision"
    assert request.cache_name == "cachedContents/same-text"


async def test_extraction_rejects_unknown_sources_and_nonliteral_quotes():
    c = claim()
    response = {
        "spans": [
            {k: c[k] for k in ["source_id", "quote", "section", "kind"]},
            dict(source_id="foreign", quote=c["quote"], section=c["section"], kind=c["kind"]),
            dict(
                source_id="ep:0:0",
                quote="La usuaria es una ingeniera profesional de Bogotá.",
                section=c["section"],
                kind=c["kind"],
            ),
        ]
    }
    compiler = HydrationCompiler(
        SimpleNamespace(generate_response=AsyncMock(return_value=response))
    )
    accepted = await compiler.extract([Source("ep:0:0", "ep", "2026-10-30", "usuario", c["quote"])])
    assert len(accepted) == 1
    assert accepted[0]["quote"] == c["quote"]
    assert accepted[0]["attribution"] == "usuario"


async def test_unknown_selection_ids_do_not_replace_model_selection_with_all_candidates(caplog):
    compiler = HydrationCompiler(
        SimpleNamespace(generate_response=AsyncMock(return_value={"ids": ["not-a-source"]}))
    )
    selected = await compiler.select([claim()])
    assert selected == []
    assert "unknown hydration selection IDs" in caplog.text


async def test_small_selection_keeps_custom_groups_and_valid_ids_without_filling_with_noise(caplog):
    compiler = HydrationCompiler(
        SimpleNamespace(
            generate_response=AsyncMock(
                return_value={
                    "ids": ["fact-1", "invented", "fact-2", "fact-1"],
                    "sections": [
                        {"id": "fact-1", "section": "  Mis mascotas\ny sus cuidados  "},
                        {"id": "fact-2", "section": "Cambio de carrera"},
                        {"id": "invented", "section": "Inventado"},
                    ],
                }
            )
        )
    )
    candidates = [{**claim(), "id": f"fact-{i}"} for i in range(16)]
    selected = await compiler.select(candidates)
    assert [c["id"] for c in selected] == ["fact-1", "fact-2"]
    assert [c["fact_section"] for c in selected] == [
        "Mis mascotas y sus cuidados",
        "Cambio de carrera",
    ]
    assert "unknown hydration selection IDs" in caplog.text


async def test_selection_reuses_previous_group_when_model_omits_or_blanks_its_title():
    compiler = HydrationCompiler(
        SimpleNamespace(
            generate_response=AsyncMock(
                return_value={"ids": ["claim"], "sections": [{"id": "claim", "section": "  "}]}
            )
        )
    )
    selected = await compiler.select([{**claim(), "fact_section": "Mi prima y sus estudios"}])
    assert selected[0]["fact_section"] == "Mi prima y sus estudios"


def test_render_preserves_third_party_and_only_deduplicates_complete_facts():
    compiler = HydrationCompiler(None, char_budget=1100)
    edges = [
        dict(
            uuid="full",
            fact="Mi prima Julia es quien estudia ingeniería en Bogotá.",
            episodes=["ep"],
            source_uuid="cousin",
            target_uuid="university",
        ),
        dict(
            uuid="partial",
            fact="Julia estudia ingeniería y se gradúa en 2028.",
            episodes=["ep"],
            source_uuid="cousin",
            target_uuid="university",
        ),
    ]
    candidates = compiler.fact_candidates([claim()], edges)
    result, retained = compiler.render([candidates[0]], edges, GraphStats(2, 2))
    assert "Mi prima Julia" in result.compilation_text
    assert "2026-10-30" in result.compilation_text
    assert result.metadata.included_edge_ids == ["full"]
    assert result.metadata.included_node_ids == []
    assert result.metadata.is_partial
    assert len(result.compilation_text) <= 1100
    assert len(retained) == 1
    # An omitted claim must never suppress its sources in RAG.
    empty, retained = HydrationCompiler(None, char_budget=10).render(
        [candidates[0]], edges, GraphStats(2, 2)
    )
    assert empty.compilation_text == ""
    assert not empty.metadata.included_edge_ids
    assert not empty.metadata.included_episode_ids
    assert not retained


async def test_hydration_reads_publication_and_preserves_v1_and_cold_v2_fallback():
    legacy = HydrationService(None)
    legacy.fetch_graph_stats = AsyncMock(return_value=GraphStats(0, 0))
    legacy._build_v1 = AsyncMock(return_value=HydrationResult("v1"))
    legacy.persisted = SimpleNamespace(read=AsyncMock(return_value=published()))
    assert (
        await legacy.build_user_knowledge("owner", "v2")
    ).compilation_text == "Current personal memory"
    legacy.fetch_graph_stats.assert_not_awaited()
    assert (await legacy.build_user_knowledge("owner", "v1")).compilation_text == "v1"


async def test_cold_read_enqueues_work_and_provider_failure_does_not_block_http():
    store = SimpleNamespace(read=AsyncMock(return_value=None), request=AsyncMock())
    service = PersistedHydration(store, None, None)
    assert await service.read("owner") is None
    store.request.assert_awaited_once_with("owner")
    store.read.side_effect = RuntimeError("database unavailable")
    assert await service.read("owner") is None


async def test_old_compilation_format_enqueues_rebuild_without_parsing_retired_payload():
    store = HydrationStore(None)
    store.current = AsyncMock(return_value={"compiler": "older-format", "result": {}})
    store.request = AsyncMock()
    store.stamp = AsyncMock()
    service = PersistedHydration(store, None, None)
    assert await service.read("owner") is None
    store.request.assert_awaited_once_with("owner")
    store.stamp.assert_not_awaited()


@pytest.mark.parametrize("selection_ids", [["claim"], ["invented"]])
async def test_worker_reuses_extraction_and_publishes_only_valid_selections(selection_ids):
    ep = dict(
        uuid="ep", source="message", valid_at="2026-10-30", content="User: " + claim()["quote"]
    )
    source = sources_from_episodes([ep])[0]
    compiler = HydrationCompiler(
        SimpleNamespace(generate_response=AsyncMock(return_value={"ids": selection_ids}))
    )
    store = SimpleNamespace(
        stamp=AsyncMock(return_value="stamp"),
        snapshot=AsyncMock(
            return_value=(
                [ep],
                [
                    dict(
                        uuid="claim",
                        fact=claim()["quote"],
                        episodes=["ep"],
                        source_uuid="cousin",
                        target_uuid="school",
                    )
                ],
            )
        ),
        cached_sources=AsyncMock(return_value={source.id: (source.hash, [claim()])}),
        prune_sources=AsyncMock(),
        save_sources=AsyncMock(),
        publish=AsyncMock(return_value=True),
        current=AsyncMock(return_value=None),
        selection_cache=lambda job: None,
    )
    legacy = SimpleNamespace(fetch_graph_stats=AsyncMock(return_value=GraphStats(0, 0)))
    service = PersistedHydration(store, compiler, legacy)
    if selection_ids == ["invented"]:
        with pytest.raises(ValueError, match="No verifiable personal memory selected"):
            await service.build(dict(user="owner", token="lease", generation=1))
        store.publish.assert_not_awaited()
        return
    assert await service.build(dict(user="owner", token="lease", generation=1))
    assert compiler.llm.generate_response.await_count == 1  # Selection only, no extraction.
    store.save_sources.assert_not_awaited()
    store.publish.assert_awaited_once()
    store.publish.reset_mock()
    store.stamp.side_effect = ["before", "after"]
    store.request = AsyncMock()
    assert not await service.build(dict(user="owner", token="lease", generation=1))
    store.publish.assert_not_awaited()
    store.request.assert_not_awaited()


async def test_graph_only_archive_keeps_legacy_memory_instead_of_publishing_empty_profile():
    store = SimpleNamespace(
        stamp=AsyncMock(return_value="stamp"),
        snapshot=AsyncMock(return_value=([], [])),
        cached_sources=AsyncMock(return_value={}),
        prune_sources=AsyncMock(),
        publish=AsyncMock(),
        current=AsyncMock(return_value=None),
        selection_cache=lambda job: None,
    )
    service = PersistedHydration(
        store,
        HydrationCompiler(None),
        SimpleNamespace(fetch_graph_stats=AsyncMock(return_value=GraphStats(2, 1))),
    )
    with pytest.raises(ValueError, match="preserving v2"):
        await service.build(dict(user="owner", token="lease", generation=1))
    store.publish.assert_not_awaited()


async def test_worker_failure_keeps_durable_work_retryable_and_releases_lease():
    job = dict(user="owner", token="lease", generation=2)
    store = SimpleNamespace(claim=AsyncMock(return_value=job), release=AsyncMock())
    service = PersistedHydration(store, None, None)
    service.build = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    assert await service.run_one()
    store.release.assert_awaited_once_with(job, failed=True, terminal=False)


async def test_store_read_rejects_edited_or_deleted_episode_without_relying_on_size():
    from app.services.hydration_compiler import COMPILER_VERSION, digest

    ep = dict(uuid="ep", source="message", valid_at="2026-10-30", content="original")
    payload = dict(
        compiler=COMPILER_VERSION,
        stamp="stamp",
        result=asdict(published()),
        source_hashes={"ep": digest(ep)},
    )
    store = HydrationStore(None)
    store.current = AsyncMock(return_value=payload)
    store.stamp = AsyncMock(return_value="stamp")
    store.rows = AsyncMock(return_value=[ep])
    store.request = AsyncMock()
    store.discard = AsyncMock()
    assert await store.read("owner")
    store.rows.return_value = [dict(ep, content="modified")]
    assert await store.read("owner") is None
    store.rows.return_value = []
    assert await store.read("owner") is None


async def test_new_revision_refreshes_existing_session_without_leaking_old_cache():
    service = HydrationService(None)
    service.persisted = SimpleNamespace(
        store=SimpleNamespace(chat_read=AsyncMock(return_value=("updated", published())))
    )
    request = ChatCompletionRequest(
        user_id="owner",
        messages=[],
        compilation="old memory",
        cache_name="cachedContents/old",
        compilationMetadata=asdict(published().metadata),
    )
    assert await service.refresh_chat_compilation(request)
    assert request.compilation == "Current personal memory"
    assert request.compilationMetadata.compilation_revision == "revision"
    assert request.cache_name is None


async def test_transient_database_failure_preserves_existing_chat_memory():
    service = HydrationService(None)
    service.persisted = SimpleNamespace(
        store=SimpleNamespace(chat_read=AsyncMock(side_effect=RuntimeError("offline")))
    )
    request = ChatCompletionRequest(
        user_id="owner",
        messages=[],
        compilation="cached memory",
        compilationMetadata=asdict(published().metadata),
        cache_name="cachedContents/active",
    )
    assert not await service.refresh_chat_compilation(request)
    assert request.compilation == "cached memory"
    assert request.cache_name == "cachedContents/active"


async def test_revoked_snapshot_and_failed_fallback_still_allow_chat_without_stale_memory():
    service = HydrationService(None)
    service.persisted = SimpleNamespace(
        store=SimpleNamespace(chat_read=AsyncMock(return_value=("revoked", None)))
    )
    service.build_user_knowledge = AsyncMock(side_effect=RuntimeError("offline"))
    request = ChatCompletionRequest(
        user_id="owner",
        messages=[],
        compilation="revoked",
        compilationMetadata=dict(
            is_partial=True,
            total_estimated_tokens=10,
            included_node_ids=[],
            included_edge_ids=[],
            compilation_revision="old",
        ),
        cache_name="cachedContents/old",
    )
    assert await service.refresh_chat_compilation(request)
    assert request.compilation is request.cache_name is None
    assert request.compilationMetadata.is_partial
    service.build_user_knowledge.assert_not_awaited()


async def test_real_hydrate_router_returns_published_text_coverage_and_legacy_contract(monkeypatch):
    monkeypatch.setattr(
        security, "get_settings", lambda: SimpleNamespace(synapse_api_secret="secret")
    )
    app = FastAPI()
    app.include_router(router)
    hydration = HydrationService(None)
    hydration.persisted = SimpleNamespace(read=AsyncMock(return_value=published()))
    app.state.hydration_service = hydration
    app.state.cache_manager = SimpleNamespace(
        create_compilation_cache=AsyncMock(return_value=(None, "small"))
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/hydrate",
            headers={"X-API-SECRET": "secret"},
            json={"userId": "owner", "version": "v2"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["userKnowledgeCompilation"] == "Current personal memory"
    assert body["compilationMetadata"]["compilation_revision"] == "revision"
    assert body["compilationMetadata"]["included_node_ids"] == []
    assert set(body["compilationMetadata"]) == {
        "is_partial",
        "total_estimated_tokens",
        "included_node_ids",
        "included_edge_ids",
        "included_episode_ids",
        "compilation_revision",
    }
    assert body["graphStats"]["entity_count"] == 1
    assert body["cacheName"] is None


async def test_ingest_notifies_after_summary_and_before_completed_job():
    # Real ingestion service ordering, with only external Graphiti and queue mocked.
    from app.schemas.models import IngestRequest
    from app.services.ingestion import IngestionService
    from app.services.job_store import create_job, get_job, remove_job

    request = IngestRequest(
        jobId="hydration-order",
        userId="owner",
        sessionId="s",
        messages=[
            dict(role="user", content="Vivo en Bogotá y trabajo como diseñadora.", timestamp=1)
        ],
        metadata=dict(sessionStartedAt=1, sessionEndedAt=2, messageCount=1),
    )
    events = []

    async def changed(user):
        assert get_job(request.jobId).status == "processing"
        events.append("queued")

    graph = SimpleNamespace(
        add_episode=AsyncMock(
            return_value=SimpleNamespace(nodes=[], edges=[], episode=SimpleNamespace(uuid="ep"))
        )
    )
    service = IngestionService(graph, "model", changed)
    service._summarize_episode = AsyncMock(return_value="Summary")

    async def store_summary(*args):
        events.append("summary")

    service._store_episode_summary = store_summary
    create_job(request.jobId, request.userId, request.sessionId)
    try:
        await service._process_background(request.jobId, request)
        assert events == ["summary", "queued"]
        assert get_job(request.jobId).status == "completed"
    finally:
        remove_job(request.jobId)
