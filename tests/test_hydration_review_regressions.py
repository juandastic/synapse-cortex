import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.schemas.models import ChatCompletionRequest, ChatMessage
from app.services.graph import GraphService
from app.services.hydration import HydrationService
from app.services.hydration_compiler import (
    COMPILER_VERSION,
    HydrationCompiler,
    UnverifiableEvidence,
    WorkPending,
    digest,
    sources_from_episodes,
)
from app.services.hydration_result import CompilationMetadata, GraphStats, HydrationResult
from app.services.hydration_store import HydrationStore, SelectionCache
from app.services.notion_correction import CorrectionItem, NotionCorrectionService
from app.services.persisted_hydration import PersistedHydration, UnsupportedArchive


def evidence(ep="a", text="Tengo diabetes. Mi gato se llama Leo."):
    return dict(
        id="quote-" + ep,
        source_id=ep + ":0:0",
        episode_id=ep,
        date="2026-11-01",
        attribution="usuario",
        quote=text,
        section="Personas y vínculos",
        kind="relato",
    )


def edge(uuid, fact, ep="a", **extra):
    return dict(
        uuid=uuid, fact=fact, episodes=[ep], source_uuid="person", target_uuid=uuid, **extra
    )


def publication():
    return HydrationResult(
        "Published memory",
        CompilationMetadata(True, 10, compilation_revision="rev"),
        GraphStats(1, 1),
    )


def request(**extra):
    return ChatCompletionRequest(
        user_id="owner",
        messages=[],
        compilation="Client memory",
        compilationMetadata=asdict(publication().metadata),
        cache_name="cachedContents/client",
        **extra,
    )


def test_invalidated_fact_never_returns_from_old_quote_even_with_other_active_edges():
    compiler = HydrationCompiler(None)
    quotes = [
        evidence(),
        dict(
            evidence("b", "No recuerdes que tengo diabetes. Era mi prima."),
            correction=True,
            attribution="corrección del usuario",
        ),
    ]
    facts = [
        edge("retired", "La usuaria tiene diabetes", active=False),
        edge("cat", "El gato se llama Leo"),
        edge("corrected", "La prima tiene diabetes", "b"),
    ]
    candidates = compiler.fact_candidates(quotes, facts)
    assert {c["id"] for c in candidates} == {"cat", "corrected"}
    result, _ = compiler.render(candidates, facts, GraphStats(2, 2))
    assert "La usuaria tiene diabetes" not in result.compilation_text
    assert "Tengo diabetes" not in result.compilation_text
    assert "La prima tiene diabetes" in result.compilation_text
    assert set(result.metadata.included_edge_ids) == {"cat", "corrected"}


@pytest.mark.parametrize("name", ["user_memory_correction", "notion_correction"])
def test_correction_is_preserved_despite_do_not_remember_words(name):
    sources = sources_from_episodes(
        [
            dict(
                uuid="correction",
                source="text",
                name=name,
                content="No recuerdes que era mi hija. Martina es mi sobrina.",
                valid_at="2026-11-01",
            )
        ]
    )
    assert len(sources) == 1
    assert sources[0].correction
    assert sources[0].attribution == "corrección del usuario"


def test_coverage_handles_graph_paraphrases_and_preserves_temporal_fields():
    compiler = HydrationCompiler(None)
    facts = [edge("lives", "The user lives in Bogota", valid_at="2020-01-01T00:00:00Z")]
    candidates = compiler.fact_candidates(
        [evidence(text="Vivo en Bogotá desde hace varios años.")], facts
    )
    result, _ = compiler.render(candidates, facts, GraphStats(1, 1))
    assert "The user lives in Bogota" in result.compilation_text
    assert "2020-01-01T00:00:00Z" in result.compilation_text
    assert result.metadata.included_edge_ids == ["lives"]
    assert result.metadata.included_node_ids == []
    assert result.metadata.included_episode_ids == ["a"]


async def test_additive_read_serves_previous_snapshot_without_revoking_or_advancing_generation():
    from app.services.hydration_compiler import digest

    ep = dict(
        uuid="a",
        source="message",
        content="User: Tengo un gato.",
        valid_at="2026-11-01",
        name="session",
    )
    payload = dict(
        compiler=COMPILER_VERSION,
        stamp="old",
        result=asdict(publication()),
        source_hashes={"a": digest(ep)},
    )
    store = HydrationStore(None)
    store.current = AsyncMock(return_value=payload)
    store.stamp = AsyncMock(return_value="new episode")
    store.rows = AsyncMock(return_value=[ep])
    store.request = AsyncMock()
    store.discard = AsyncMock()
    result = await store.read("owner")
    assert result.compilation_text == "Published memory"
    store.discard.assert_not_awaited()
    store.request.assert_awaited_once_with("owner")


async def test_destructive_read_discards_only_the_checked_snapshot():
    ep = dict(uuid="a", source="message", content="old", valid_at="2026-11-01")
    store = HydrationStore(None)
    store.current = AsyncMock(
        return_value=dict(
            compiler=COMPILER_VERSION,
            stamp="stamp",
            result=asdict(publication()),
            source_hashes={"a": digest(ep)},
        )
    )
    store.stamp = AsyncMock(return_value="stamp")
    store.rows = AsyncMock(return_value=[])
    store.discard = AsyncMock()
    store.request = AsyncMock()
    assert await store.read("owner") is None
    store.discard.assert_awaited_once_with("owner", "rev")
    store.request.assert_not_awaited()


@pytest.mark.parametrize("mode", ["no_memory", "v1", "legacy"])
async def test_chat_does_not_enable_memory_for_opted_out_or_legacy_clients(mode):
    store = SimpleNamespace(chat_read=AsyncMock())
    service = HydrationService(None)
    service.persisted = SimpleNamespace(store=store)
    req = request()
    if mode == "no_memory":
        req.compilation = None
    elif mode == "v1":
        req.compilationMetadata = None
    else:
        req.compilation = None
        req.messages = [ChatMessage(role="system", content="Legacy inline personal memory")]
    before = req.model_dump()
    assert not await service.refresh_chat_compilation(req)
    assert req.model_dump() == before
    store.chat_read.assert_not_awaited()


async def test_chat_timeout_keeps_client_memory_without_building_v2_or_creating_cache():
    async def slow(*args):
        await asyncio.Event().wait()

    service = HydrationService(None)
    service.persisted = SimpleNamespace(store=SimpleNamespace(chat_read=slow))
    service.build_user_knowledge = AsyncMock()
    req = request()
    assert not await service.refresh_chat_compilation(req)
    assert req.compilation == "Client memory"
    assert req.cache_name == "cachedContents/client"
    service.build_user_knowledge.assert_not_awaited()


async def test_argon_refresh_never_uses_the_prepared_explicit_cache():
    new = publication()
    new.provider_cache_name = "cachedContents/prepared"
    new.provider_cache_model = "gemini-3.1-pro-preview"
    service = HydrationService(None)
    service.persisted = SimpleNamespace(
        store=SimpleNamespace(chat_read=AsyncMock(return_value=("updated", new)))
    )
    req = request(model="gemini-4-argon")
    assert await service.refresh_chat_compilation(req)
    assert req.compilation == "Published memory"
    assert req.cache_name is None


async def test_selector_reuses_cached_decisions_and_yields_before_third_uncached_call():
    store = SimpleNamespace(rows=AsyncMock(return_value=[]))
    cache = SelectionCache(store, dict(user="owner", token="lease", generation=1))
    assert await cache.load("one") is None
    assert await cache.load("two") is None
    with pytest.raises(WorkPending):
        await cache.load("three")
    store.rows.return_value = [dict(ids=["fact"])]
    assert await cache.load("already-cached") == ["fact"]
    with pytest.raises(WorkPending):
        await cache.load("legacy-needs-classification", require_sections=True)


async def test_incremental_selection_uses_previous_context_plus_changed_facts_only():
    ep = dict(
        uuid="a", source="message", valid_at="2026-11-01", content="User: " + evidence()["quote"]
    )
    source = sources_from_episodes([ep])[0]
    compiler = HydrationCompiler(
        SimpleNamespace(generate_response=AsyncMock(return_value=dict(ids=["old0", "new"])))
    )
    oldedges = [edge(f"old{i}", f"Personal fact {i}") for i in range(100)]
    old_candidates = compiler.fact_candidates([evidence()], oldedges)
    previous = dict(
        compiler=COMPILER_VERSION,
        selected=old_candidates[:3],
        seen_hashes={c["id"]: digest(c) for c in old_candidates},
    )
    store = SimpleNamespace(
        stamp=AsyncMock(return_value="stamp"),
        snapshot=AsyncMock(
            return_value=([ep], oldedges + [edge("new", "A newly mentioned project")])
        ),
        prune_sources=AsyncMock(),
        cached_sources=AsyncMock(return_value={source.id: (source.hash, [evidence()])}),
        current=AsyncMock(return_value=previous),
        selection_cache=lambda job: None,
        publish=AsyncMock(return_value=True),
    )
    service = PersistedHydration(
        store,
        compiler,
        SimpleNamespace(_fetch_graph_stats=AsyncMock(return_value=GraphStats(2, 101))),
    )
    assert await service.build(dict(user="owner", token="lease", generation=1))
    import json

    offered = json.loads(compiler.llm.generate_response.await_args.args[0][1].content)
    assert {c["id"] for c in offered} == {"old0", "old1", "old2", "new"}


async def test_unsupported_archive_becomes_terminal_until_memory_changes():
    job = dict(user="owner", token="lease", generation=1)
    store = SimpleNamespace(claim=AsyncMock(return_value=job), release=AsyncMock())
    service = PersistedHydration(store, None, None)
    service.build = AsyncMock(side_effect=UnsupportedArchive())
    assert await service.run_one()
    store.release.assert_awaited_once_with(job, failed=False, terminal=True)


async def test_heartbeat_survives_one_transient_database_failure(monkeypatch):
    original_sleep = asyncio.sleep
    renewed = asyncio.Event()
    calls = 0

    async def fast_sleep(_):
        await original_sleep(0)

    async def renew(job):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary database outage")
        renewed.set()
        return True

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)
    service = PersistedHydration(SimpleNamespace(renew=renew), None, None)
    build = asyncio.create_task(asyncio.Event().wait())
    lost = asyncio.Event()
    heartbeat = asyncio.create_task(service._heartbeat({}, build, lost))
    try:
        await asyncio.wait_for(renewed.wait(), timeout=1)
        assert calls >= 2 and not build.done() and not lost.is_set()
    finally:
        heartbeat.cancel()
        build.cancel()
        await asyncio.gather(heartbeat, build, return_exceptions=True)


async def test_render_uses_fact_classification_and_never_inherits_unrelated_user_labels():
    quote = dict(
        evidence(text="Prefiero respuestas breves."),
        section="Preferencias de interacción",
        kind="preferencia",
    )
    compiler = HydrationCompiler(
        SimpleNamespace(
            generate_response=AsyncMock(
                return_value={
                    "ids": ["medication"],
                    "sections": {"medication": "Situación reciente"},
                }
            )
        )
    )
    facts = [edge("medication", "La prima toma el medicamento X")]
    selected = await compiler.select(compiler.fact_candidates([quote], facts))
    result, _ = compiler.render(selected, facts, GraphStats(2, 1))
    assert "Situación reciente" in result.compilation_text
    assert "Preferencias de interacción" not in result.compilation_text
    assert "; usuario;" not in result.compilation_text
    assert "; preferencia;" not in result.compilation_text
    assert "conversación/documento" in result.compilation_text
    # No classifier result is safer than borrowing the episode's label.
    selected[0].pop("fact_section")
    unclassified, _ = compiler.render(selected, facts, GraphStats(2, 1))
    assert "Registros sin clasificar" in unclassified.compilation_text
    compiler.llm.generate_response.return_value = {
        "ids": ["medication"],
        "sections": {"medication": "Unrecognized category"},
    }
    selected = await compiler.select(compiler.fact_candidates([quote], facts))
    neutral, _ = compiler.render(selected, facts, GraphStats(2, 1))
    assert "Registros sin clasificar" in neutral.compilation_text
    assert "Unrecognized category" not in neutral.compilation_text


async def test_poison_extraction_is_quarantined_and_yields_without_blocking_other_batches():
    ep = dict(uuid="a", source="message", valid_at="2026-11-01", content="User: Vivo en Bogotá.")
    source = sources_from_episodes([ep])[0]
    store = SimpleNamespace(
        stamp=AsyncMock(return_value="stamp"),
        snapshot=AsyncMock(return_value=([ep], [])),
        prune_sources=AsyncMock(),
        cached_sources=AsyncMock(return_value={}),
        save_sources=AsyncMock(),
    )
    compiler = SimpleNamespace(extract=AsyncMock(side_effect=UnverifiableEvidence()))
    service = PersistedHydration(store, compiler, None)
    job = dict(user="owner", token="lease", generation=1)
    with pytest.raises(WorkPending):
        await service.build(job)
    store.save_sources.assert_awaited_once_with(
        job, [source], [], quarantine_reason="nonliteral_quotes"
    )


@pytest.mark.parametrize("kind", ["explorer", "notion"])
async def test_hydration_outage_does_not_prevent_classic_memory_correction(kind):
    graph = SimpleNamespace(add_episode=AsyncMock(return_value=SimpleNamespace(nodes=[], edges=[])))
    callback = AsyncMock(side_effect=RuntimeError("temporary hydration store failure"))
    if kind == "explorer":
        await GraphService(None, graph, callback).correct_memory("owner", "Martina es mi sobrina.")
    else:
        service = NotionCorrectionService(graph, None, callback)
        item = CorrectionItem("Familia", "db", "page", "Martina", {}, {}, "Es mi sobrina.")
        await service._correct_graph(item, "owner", "Spanish")
    graph.add_episode.assert_awaited_once()
    assert callback.await_count == 2
