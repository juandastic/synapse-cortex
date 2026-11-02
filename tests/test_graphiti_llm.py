import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.prompts.models import Message
from tenacity import wait_none

from app.core import posthog
from app.core.config import HYDRATION_MODEL, Settings
from app.core.graphiti_client import create_graphiti
from app.core.graphiti_llm import OpenRouterGraphitiClient
from app.core.openrouter_usage import openrouter_usage_properties
from app.core.posthog import ContextualGeminiModels, get_llm_context, posthog_user_context
from app.services.hydration_compiler import HydrationCompiler, Source
from app.services.hydration_result import GraphStats
from app.services.ingestion import EpisodeSummary, IngestionService


def completion(content, *, model="openai/test-model", usage=None, finish_reason="stop"):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": json.dumps(content)},
            }
        ],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


@pytest.fixture
def analytics(monkeypatch):
    client = Mock()
    monkeypatch.setattr(posthog, "_posthog_client", client)
    return client


@pytest.fixture
async def make_client():
    clients = []

    def make(handler, model="openai/main"):
        sdk = openai.AsyncOpenAI(
            api_key="test-only",
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        client = OpenRouterGraphitiClient(
            LLMConfig(model=model, small_model="openai/small"), client=sdk
        )
        clients.append(client)
        return client

    yield make
    for client in clients:
        await client.close()


async def test_graphiti_roles_use_strict_schemas_and_report_actual_usage(make_client, analytics):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json=completion(
                {
                    "extracted_entities": [
                        {"name": "Ana", "entity_type_id": 0, "episode_indices": [0]}
                    ]
                },
                model=payload["model"] + "-snapshot",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cost": 0.00002,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                    "prompt_tokens_details": {"cached_tokens": 30},
                },
            ),
        )

    client = make_client(handler)
    messages = [
        Message(role="system", content="Extract entities"),
        Message(role="user", content="Ana"),
    ]
    for size in (ModelSize.medium, ModelSize.small):
        with posthog_user_context("owner", "trace", "session"):
            result = await client.generate_response(
                messages,
                response_model=ExtractedEntities,
                model_size=size,
                prompt_name="extract_nodes.extract_message",
            )
        assert result["extracted_entities"][0]["name"] == "Ana"
    assert messages[0].content == "Extract entities"
    assert [request["model"] for request in requests] == ["openai/main", "openai/small"]
    assert [request["reasoning"]["effort"] for request in requests] == ["low", "none"]
    for request in requests:
        assert request["provider"]["require_parameters"] is True
        assert "temperature" not in request
        schema = request["response_format"]["json_schema"]
        assert schema["strict"] is True
        assert schema["schema"]["additionalProperties"] is False
        # Defaulted fields in Graphiti's real schemas also become required in strict JSON.
        entity_schema = schema["schema"]["$defs"]["ExtractedEntity"]
        assert "episode_indices" in entity_schema["required"]
    assert analytics.capture.call_count == 2
    for call in analytics.capture.call_args_list:
        event = call.kwargs
        properties = event["properties"]
        assert event["distinct_id"] == "owner"
        assert properties["$ai_trace_id"] == "trace"
        assert properties["$ai_session_id"] == "session"
        assert properties["$ai_provider"] == "openrouter"
        assert properties["$ai_model"].endswith("-snapshot")
        assert properties["$ai_input_tokens"] == 100
        assert properties["$ai_output_tokens"] == 20
        assert properties["$ai_reasoning_tokens"] == 5
        assert properties["$ai_cache_read_input_tokens"] == 30
        assert properties["$ai_total_cost_usd"] == 0.00002
        assert 0 <= properties["$ai_latency"] < 5  # seconds, not milliseconds
        assert properties["prompt_name"] == "extract_nodes.extract_message"


async def test_overlapping_users_keep_their_session_and_prompt(make_client, analytics):
    entered = 0
    both_entered = asyncio.Event()

    async def handler(request):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), 2)
        payload = json.loads(request.content)
        return httpx.Response(200, json=completion({"summary": payload["messages"][-1]["content"]}))

    client = make_client(handler)

    async def summarize(owner):
        with posthog_user_context(owner, f"trace-{owner}", f"session-{owner}"):
            return await asyncio.create_task(
                client.generate_response(
                    [Message(role="user", content=owner)],
                    response_model=EpisodeSummary,
                    prompt_name=f"summary-{owner}",
                )
            )

    await asyncio.gather(summarize("alice"), summarize("bob"))
    assert get_llm_context() is None
    for call in analytics.capture.call_args_list:
        owner = call.kwargs["distinct_id"]
        properties = call.kwargs["properties"]
        assert properties["$ai_session_id"] == f"session-{owner}"
        assert properties["$ai_trace_id"] == f"trace-{owner}"
        assert properties["prompt_name"] == f"summary-{owner}"
        assert json.loads(properties["$ai_output_choices"][0]["content"])["summary"] == owner


async def test_failed_extraction_is_reported_and_does_not_become_empty_success(
    make_client, analytics
):
    client = make_client(lambda _: httpx.Response(401, json={"error": {"message": "Invalid key"}}))
    with posthog_user_context("owner", "trace", "session"):
        with pytest.raises(httpx.HTTPStatusError):
            await client.generate_response(
                [Message(role="user", content="Extract")], response_model=ExtractedEntities
            )
    assert analytics.capture.call_count == 1
    properties = analytics.capture.call_args.kwargs["properties"]
    assert properties["$ai_is_error"] is True
    assert properties["$ai_session_id"] == "session"
    assert "$ai_input_tokens" not in properties


async def test_analytics_failure_does_not_fail_extraction(make_client, analytics):
    analytics.capture.side_effect = RuntimeError("analytics unavailable")
    client = make_client(lambda _: httpx.Response(200, json=completion({"summary": "Useful fact"})))
    assert await client.generate_response(
        [Message(role="user", content="Summarize")], response_model=EpisodeSummary
    ) == {"summary": "Useful fact"}


async def test_truncated_response_preserves_billable_tokens(make_client, analytics):
    client = make_client(lambda _: httpx.Response(200, json=completion({}, finish_reason="length")))
    with pytest.raises(openai.LengthFinishReasonError):
        await client.generate_response(
            [Message(role="user", content="Extract")], response_model=ExtractedEntities
        )
    properties = analytics.capture.call_args.kwargs["properties"]
    assert properties["$ai_is_error"] is True
    assert properties["$ai_input_tokens"] == 10


@pytest.mark.parametrize("status", [429, 503, "connection"])
async def test_transient_attempts_are_retried_and_tracked_once_each(
    make_client, analytics, monkeypatch, status
):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if status == "connection":
                raise httpx.ConnectError("connection unavailable", request=request)
            return httpx.Response(status, json={"error": {"message": "Try again"}})
        return httpx.Response(200, json=completion({"summary": "Recovered"}))

    client = make_client(handler)
    monkeypatch.setattr(client._generate_response_with_retry.retry, "wait", wait_none())
    with posthog_user_context("owner", "trace", "session"):
        assert await client.generate_response(
            [Message(role="user", content="Summarize")], response_model=EpisodeSummary
        ) == {"summary": "Recovered"}
    assert attempts == analytics.capture.call_count == 2
    assert analytics.capture.call_args_list[0].kwargs["properties"]["$ai_is_error"] is True
    assert "$ai_is_error" not in analytics.capture.call_args_list[1].kwargs["properties"]
    assert all(
        call.kwargs["properties"]["$ai_session_id"] == "session"
        for call in analytics.capture.call_args_list
    )


@pytest.mark.parametrize(
    "usage,expected",
    [
        ({"cost": 0.01}, 0.01),
        ({"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.01}}, 0.01),
        (
            {"cost": 0.001, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.01}},
            0.011,
        ),
        ({"cost": 0.01, "is_byok": False, "cost_details": {"upstream_inference_cost": 0.01}}, 0.01),
        ({"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0}}, 0),
    ],
)
def test_byok_cost_includes_provider_bill_without_double_counting_credits(usage, expected):
    properties = openrouter_usage_properties(usage)
    assert properties["$ai_total_cost_usd"] == pytest.approx(expected)
    assert properties["openrouter_cost_usd"] == usage["cost"]


async def test_graph_generation_captures_byok_provider_cost(make_client, analytics):
    client = make_client(
        lambda _: httpx.Response(
            200,
            json=completion(
                {"summary": "Fact"},
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "cost": 0,
                    "is_byok": True,
                    "cost_details": {"upstream_inference_cost": 0.01},
                },
            ),
        )
    )
    await client.generate_response(
        [Message(role="user", content="Summarize")], response_model=EpisodeSummary
    )
    properties = analytics.capture.call_args.kwargs["properties"]
    assert properties["$ai_total_cost_usd"] == 0.01
    assert properties["openrouter_cost_usd"] == 0
    assert properties["is_byok"] is True


async def test_summary_uses_small_graph_client_and_preserves_context(make_client, analytics):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == "openai/small"
        assert len(payload["messages"][-1]["content"]) == 8000
        return httpx.Response(200, json=completion({"summary": "  Lives in Bogota.  "}))

    service = IngestionService(SimpleNamespace(llm_client=make_client(handler)), "openai/main")
    with posthog_user_context("owner", "trace", "session"):
        assert await service._summarize_episode("long conversation " * 1000) == "Lives in Bogota."
    properties = analytics.capture.call_args.kwargs["properties"]
    assert properties["$ai_session_id"] == "session"
    assert properties["prompt_name"] == "ingestion.episode_summary"


async def test_vertex_fallback_also_keeps_overlapping_contexts_isolated():
    calls = []

    async def generate(**kwargs):
        await asyncio.sleep(0)
        calls.append(kwargs)

    original = SimpleNamespace(
        generate_content=generate, _default_distinct_id="unchanged", _default_properties={}
    )
    models = ContextualGeminiModels(original)

    async def call(owner):
        with posthog_user_context(owner, f"trace-{owner}", f"session-{owner}"):
            await models.generate_content(model="gemini-test")

    await asyncio.gather(call("alice"), call("bob"))
    assert original._default_distinct_id == "unchanged"
    assert original._default_properties == {}
    for call in calls:
        owner = call["posthog_distinct_id"]
        assert call["posthog_trace_id"] == f"trace-{owner}"
        assert call["posthog_properties"]["$ai_session_id"] == f"session-{owner}"


@pytest.mark.parametrize("missing", ["key", "model"])
def test_factory_rejects_incomplete_openrouter_configuration(missing):
    settings = Settings(
        _env_file=None,
        neo4j_password="test",
        synapse_api_secret="test",
        graphiti_provider="openrouter",
        graphiti_model="openai/gpt-6-luna",
        graphiti_small_model="openai/gpt-6-luna",
        open_router_api_key="test" if missing != "key" else "",
    )
    if missing == "model":
        settings.graphiti_small_model = "gemini-2.5-flash-lite"
    with pytest.raises(ValueError, match="OPEN_ROUTER_API_KEY|OpenAI model IDs"):
        create_graphiti(settings, Mock())


async def test_factory_preserves_embeddings_and_vertex_rollback(monkeypatch):
    # Mock only database construction, keep the real factory and provider adapters.
    monkeypatch.setattr(
        "graphiti_core.graphiti.Neo4jDriver", Mock(return_value=Mock(spec=GraphDriver))
    )
    settings = Settings(
        _env_file=None,
        neo4j_password="test",
        synapse_api_secret="test",
        graphiti_provider="vertex",
        graphiti_model="gemini-test",
        graphiti_small_model="gemini-small",
    )
    google_client = Mock()
    graph = create_graphiti(settings, google_client)
    assert graph.embedder.config.embedding_model == "gemini-embedding-001"
    assert graph.embedder.config.embedding_dim == 1024
    assert graph.llm_client.small_model == "gemini-small"
    settings.graphiti_provider = "openrouter"
    settings.graphiti_model = settings.graphiti_small_model = "openai/gpt-6-luna"
    settings.open_router_api_key = "test"
    graph = create_graphiti(settings, google_client)
    try:
        assert graph.embedder.client is google_client
        assert isinstance(graph.cross_encoder, GeminiRerankerClient)
        assert graph.cross_encoder.client is google_client
        assert graph.cross_encoder.config.model is None  # Google's default, never an OpenAI ID.
    finally:
        await graph.llm_client.close()


async def test_hydration_uses_fixed_model_with_existing_openrouter_tracking(make_client, analytics):
    requests = []
    source = Source("ep:0:0", "ep", "2026-11-01", "usuario", "Martina es mi sobrina.")

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        schema = payload["response_format"]["json_schema"]
        if schema["name"] == "Extraction":
            answer = {
                "spans": [
                    {
                        "source_id": source.id,
                        "quote": source.text,
                        "section": "Personas y vínculos",
                        "kind": "relato",
                    }
                ]
            }
        else:
            answer = {
                "ids": ["fact"],
                "sections": [{"id": "fact", "section": "Personas y vínculos"}],
            }
        return httpx.Response(
            200,
            json=completion(
                answer,
                model=payload["model"],
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cost": 0.001,
                },
            ),
        )

    hydration_client = make_client(handler, model=HYDRATION_MODEL)
    compiler = HydrationCompiler(hydration_client)
    facts = [dict(uuid="fact", fact=source.text, episodes=["ep"])]
    with posthog_user_context("owner", "hydration-trace", "session"):
        quotes = await compiler.extract([source])
        selected = await compiler.select(compiler.fact_candidates(quotes, facts))
    result, _ = compiler.render(selected, facts, GraphStats(2, 1))
    assert "Martina es mi sobrina" in result.compilation_text
    assert result.metadata.included_edge_ids == ["fact"]
    assert len(requests) == 2

    def assert_strict_objects(schema):
        if isinstance(schema, dict):
            if schema.get("type") == "object":
                assert schema["additionalProperties"] is False
            for item in schema.values():
                assert_strict_objects(item)
        elif isinstance(schema, list):
            for item in schema:
                assert_strict_objects(item)

    for request in requests:
        assert request["model"] == HYDRATION_MODEL
        assert 0 < request["max_completion_tokens"] <= 16384
        assert request["reasoning"]["effort"] == "low"
        assert request["response_format"]["json_schema"]["strict"] is True
        assert_strict_objects(request["response_format"]["json_schema"]["schema"])
    assert analytics.capture.call_count == 2
    names = set()
    for call in analytics.capture.call_args_list:
        assert call.kwargs["distinct_id"] == "owner"
        props = call.kwargs["properties"]
        assert props["$ai_trace_id"] == "hydration-trace"
        assert props["$ai_session_id"] == "session"
        assert props["$ai_provider"] == "openrouter"
        assert props["$ai_total_cost_usd"] == 0.001
        names.add(props["prompt_name"])
    assert names == {"hydration.extract", "hydration.select"}


async def test_hydration_limits_provider_call_duration(monkeypatch):
    monkeypatch.setattr("app.services.hydration_compiler.LLM_TIMEOUT_SECONDS", 0.01)

    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    compiler = HydrationCompiler(SimpleNamespace(generate_response=stalled))
    source = Source("ep:0:0", "ep", "2026-11-01", "usuario", "Martina es mi sobrina.")
    with pytest.raises(TimeoutError):
        await compiler.extract([source])
