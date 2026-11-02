import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.schemas.models import ChatCompletionRequest, ChatMessage
from app.services.graph import GraphService
from app.services.hydration import HydrationService
from app.services.hydration_result import CompilationMetadata, GraphStats, HydrationResult
from app.services.notion_correction import CorrectionItem, NotionCorrectionService
from app.services.persisted_hydration import PersistedHydration, UnsupportedArchive


def episode(ep="a"):
    return dict(uuid=ep, valid_at="2026-11-01", source="message", name="session")


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
