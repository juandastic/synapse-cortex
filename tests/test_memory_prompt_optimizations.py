import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.api.routes import _compilation_cache, _prepare_chat_cache, chat_completions
from app.core.config import Settings
from app.schemas.models import ChatCompletionRequest, ChatMessage, CompilationMetadataResponse
from app.services.cache_manager import MIN_CHARS_FOR_CACHE, CacheManager
from app.services.generation import GenerationService
from app.services.hydration import HydrationService
from app.services.hydration_compiler import HydrationCompiler, fact_hash
from app.services.hydration_consolidation import apply_plan
from app.services.hydration_result import GraphStats
from app.services.hydration_store import HydrationStore
from app.services.openrouter_generation import OpenRouterGenerationService
from app.services.prompt_context import RAG_HEADER


def request(**overrides):
    return ChatCompletionRequest(
        **{
            "model": "gemini-4-argon",
            "user_id": "owner",
            "system_instruction": "Persona\n\nCurrent date and time: Thursday 12:00 UTC",
            "compilation": "Julia es mi prima.",
            "compilationMetadata": CompilationMetadataResponse(
                is_partial=True,
                total_estimated_tokens=10,
                included_edge_ids=["known"],
                included_node_ids=[],
            ),
            "messages": [{"role": "user", "content": "¿Y Julia?"}],
            **overrides,
        }
    )


@pytest.mark.parametrize("legacy", [False, True])
async def test_variable_clock_and_retrieval_leave_the_same_persona_and_core_prefix(legacy):
    gemini = GenerationService(None)
    async with httpx.AsyncClient() as client:
        alternative = OpenRouterGenerationService("test", client)
        for clock, fact in [("12:00", "likes hiking"), ("12:01", "has a cat")]:
            instructions = (
                f"Persona\n\nCurrent date and time: Thursday {clock} UTC{RAG_HEADER}{fact}"
            )
            req = request(system_instruction=instructions)
            if legacy:
                req.system_instruction = None
                req.messages.insert(0, ChatMessage(role="system", content=instructions))
            contents = await gemini._build_contents(req, inline_compilation=True)
            gemini_text = contents[0].parts[0].text
            req.provider = "openrouter"
            req.model = "openai/gpt-6.1-sol"
            other_text = alternative.build_payload(req)["messages"][0]["content"]
            for text in (gemini_text, other_text):
                assert text.startswith("Persona\n\nJulia es mi prima.\n\n")
                assert clock in text and fact in text
                assert text.count("Julia es mi prima.") == 1
            assert contents[0].parts[-1].text == "¿Y Julia?"


async def test_explicit_cache_does_not_duplicate_core_and_keeps_variable_context():
    req = request(system_instruction=f"Persona{RAG_HEADER}Fresh memory")
    contents = await GenerationService(None)._build_contents(req, inline_compilation=False)
    text = contents[0].parts[0].text
    assert "Persona" in text and "Fresh memory" in text
    assert "Julia es mi prima." not in text


@pytest.mark.parametrize("provider", ["vertex", "openrouter"])
async def test_slow_graph_is_cancelled_and_router_still_generates_with_core(monkeypatch, provider):
    monkeypatch.setattr("app.services.graph_rag.RAG_TIMEOUT_SECONDS", 0.01)
    cancelled = asyncio.Event()

    async def slow_search(*args, **kwargs):
        assert kwargs["group_ids"] == ["owner"]
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    class Generator:
        def validate_request(self, req):
            pass

        async def stream_chat_completion(self, req, **kwargs):
            assert cancelled.is_set()
            assert req.compilation == "Julia es mi prima."
            assert req.system_instruction.startswith("Persona")
            assert req.rag_usage_fields == {"rag_enabled": False}
            yield "answer with core"

    req = request(provider=provider)
    response = await chat_completions(
        req,
        "test",
        Generator(),
        Generator(),
        SimpleNamespace(search_=slow_search),
        object(),
        HydrationService(None),
    )
    assert [chunk async for chunk in response.body_iterator] == ["answer with core"]


def test_compact_render_keeps_literal_facts_evidence_and_original_timestamps():
    rows = [
        dict(
            uuid="fact",
            fact="Julia es mi prima, no mi hija.",
            source_uuid="owner",
            target_uuid="Julia",
            date="2026-11-05T18:43:51.825423Z",
            date_kind="registro",
            valid_at="2026-11-02T13:10:11.123Z",
            invalid_at=None,
            episode_ids=["episode"],
        )
    ]
    before = fact_hash(rows[0])
    compiler = HydrationCompiler(None)
    candidates = compiler.fact_candidates(rows)
    result, retained = compiler.render(apply_plan(candidates, {}), rows, GraphStats(2, 1))
    assert (
        "[registro 2026-11-05] Julia es mi prima, no mi hija. [desde 2026-11-02]"
        in result.compilation_text
    )
    assert "; conversación/documento" not in result.compilation_text
    assert result.metadata.included_edge_ids == ["fact"]
    assert result.metadata.included_episode_ids == ["episode"]
    assert retained[0]["date"] == rows[0]["date"]
    assert retained[0]["valid_at"] == rows[0]["valid_at"]
    assert fact_hash(rows[0]) == before


def test_intraday_validity_does_not_look_like_an_empty_date_range():
    row = dict(
        uuid="fact",
        fact="Plan temporal.",
        date="2026-11-05",
        valid_at="2026-11-06T08:00:00Z",
        invalid_at="2026-11-06T10:00:00Z",
    )
    compiler = HydrationCompiler(None)
    result, _ = compiler.render(compiler.fact_candidates([row]), [row], GraphStats(2, 1))
    assert "[desde 2026-11-06T08:00:00Z] [hasta 2026-11-06T10:00:00Z]" in result.compilation_text


async def test_cache_is_prepared_only_for_an_actual_legacy_chat():
    settings = Settings(_env_file=None, neo4j_password="test", synapse_api_secret="test")
    create = AsyncMock(return_value=SimpleNamespace(name="cachedContents/legacy"))
    client = SimpleNamespace(aio=SimpleNamespace(caches=SimpleNamespace(create=create)))
    manager = CacheManager(client, settings.chat_model)
    hydration = HydrationService(None)
    from app.services.hydration_result import HydrationResult

    result = HydrationResult("x" * MIN_CHARS_FOR_CACHE)
    assert await _compilation_cache("owner", result, manager, hydration) == (
        None,
        "deferred_until_chat",
    )
    for provider, model in [("vertex", "gemini-4-argon"), ("openrouter", "openai/gpt-6.1-sol")]:
        req = request(provider=provider, model=model, compilation=result.compilation_text)
        await _prepare_chat_cache(req, manager, hydration)
        assert req.cache_name is None
    create.assert_not_awaited()
    req = request(model="gemini-3.1-pro-preview", compilation=result.compilation_text)
    await _prepare_chat_cache(req, manager, hydration)
    assert req.cache_name == "cachedContents/legacy"
    await _prepare_chat_cache(req, manager, hydration)
    create.assert_awaited_once()
    assert create.await_args.kwargs["model"] == "gemini-3.1-pro-preview"


@pytest.mark.parametrize("failure", ["error", "timeout"])
async def test_legacy_cache_failure_keeps_memory_inline(monkeypatch, failure):
    monkeypatch.setattr("app.api.routes.CACHE_PREPARATION_TIMEOUT_SECONDS", 0.01)

    async def unavailable(*args):
        if failure == "timeout":
            await asyncio.Event().wait()
        raise RuntimeError("cache unavailable")

    manager = SimpleNamespace(
        model_name="gemini-3.1-pro-preview", create_compilation_cache=unavailable
    )
    req = request(model="gemini-3.1-pro-preview")
    await _prepare_chat_cache(req, manager, HydrationService(None))
    assert req.cache_name is None
    assert req.compilation == "Julia es mi prima."


async def test_legacy_chat_reuses_published_cache_across_fresh_worker_managers():
    store = HydrationStore(None)
    store.rows = AsyncMock(
        side_effect=[
            [],
            [],
            [
                {
                    "name": "cachedContents/shared",
                    "model": "gemini-3.1-pro-preview",
                    "expires": time.time() + 900,
                }
            ],
        ]
    )
    hydration = HydrationService(None)
    hydration.persisted = SimpleNamespace(store=store)
    creates = []
    for _ in range(2):
        create = AsyncMock(return_value=SimpleNamespace(name="cachedContents/shared"))
        creates.append(create)
        client = SimpleNamespace(aio=SimpleNamespace(caches=SimpleNamespace(create=create)))
        manager = CacheManager(client, "gemini-3.1-pro-preview")
        req = request(model="gemini-3.1-pro-preview", compilation="x" * MIN_CHARS_FOR_CACHE)
        req.compilationMetadata.compilation_revision = "revision"
        await _prepare_chat_cache(req, manager, hydration)
        assert req.cache_name == "cachedContents/shared"
    creates[0].assert_awaited_once()
    creates[1].assert_not_awaited()
    assert store.rows.await_args.kwargs == {"user": "owner", "revision": "revision"}
