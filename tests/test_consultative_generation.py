"""Second opinions preserve author ownership, public stream contracts and fail-open chat."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import router
from app.core import security
from app.schemas.consultation import ConsultationAdvice, ConsultationSource, RouteDecision
from app.schemas.models import ChatCompletionChunk, ChatCompletionRequest, ChatMessage
from app.services import consultative_generation
from app.services.consultative_generation import ConsultativeGenerationService
from app.services.graph_rag import GraphRagOutcome
from app.services.medical_consultant import ConsultationError

DONE = "data: [DONE]\n\n"
DRAFT = "PRIVATE draft: the supplement never causes headaches."
ADVICE = "PRIVATE advice: qualify certainty and consider individual caffeine sensitivity."
FINAL = "Final author answer: consider sensitivity and the dose on the label."


def request(provider="vertex"):
    return ChatCompletionRequest(
        provider=provider,
        model="gemini-4-argon" if provider == "vertex" else "openai/gpt-6.1-sol",
        system_instruction="Persona: supportive Spanish-speaking companion. Retrieved: headaches.",
        compilation="User-reported history: sensitive to caffeine; unverified interpretation.",
        cache_name="cachedContents/example",
        messages=[
            {"role": "user", "content": "Estoy evaluando un suplemento de cafeína y teanina."},
            {"role": "assistant", "content": "Comparemos la dosis y tu experiencia."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "¿Este?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,bGFiZWw="},
                    },
                ],
            },
        ],
        user_id="test-owner",
        session_id="test-session",
        posthog_trace_id="test-trace",
    )


def event(req, *, text=None, usage=None, error=None, finish=None):
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": req.model,
        "provider": req.provider,
        "choices": [],
    }
    if text is not None or finish is not None:
        chunk["choices"] = [
            {
                "index": 0,
                "delta": {"content": text} if text is not None else {},
                "finish_reason": finish,
            }
        ]
    if usage is not None:
        chunk["usage"] = usage
    if error is not None:
        chunk["error"] = {"message": error}
    return f"data: {json.dumps(chunk)}\n\n"


def usage(prompt=10, completion=2, cost=0.01, **extra):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cost": cost,
        **extra,
    }


def successful_events(req, text, reported_usage=None):
    return [
        event(req, text=text),
        event(req, usage=reported_usage or usage(), finish="stop"),
        DONE,
    ]


def data_chunks(events):
    return [
        json.loads(line.removeprefix("data:").strip())
        for entry in events
        for line in entry.splitlines()
        if line.startswith("data:") and line.removeprefix("data:").strip() != "[DONE]"
    ]


def visible_text(events):
    return "".join(
        choice.get("delta", {}).get("content") or ""
        for chunk in data_chunks(events)
        for choice in chunk.get("choices", [])
    )


def final_usage(events):
    reported = [chunk["usage"] for chunk in data_chunks(events) if chunk.get("usage")]
    assert len(reported) == 1, "A public answer must report its combined usage exactly once"
    return reported[0]


class Author:
    """External author double, preserving real provider SSE boundaries and failures."""

    def __init__(self, scripts=None):
        self.scripts = scripts or {}
        self.requests = []
        self.cache_managers = []
        self.closed_stages = []
        self.validate_request = Mock()

    async def stream_chat_completion(self, req, *, cache_manager=None):
        self.requests.append(req.model_copy(deep=True))
        self.cache_managers.append(cache_manager)
        try:
            script = self.scripts.get(req.generation_stage)
            if script is None:
                text = {"draft": DRAFT, "final": FINAL}.get(req.generation_stage, FINAL)
                script = successful_events(req, text)
            elif callable(script):
                script = script(req)
            for item in script:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self.closed_stages.append(req.generation_stage)


def consultant(category="medical", *, routing_usage=None, advice_usage=None):
    return SimpleNamespace(
        timeout_seconds=1.0,
        routing_timeout_seconds=1.0,
        classify=AsyncMock(
            return_value=RouteDecision(
                category=category,
                probability=0.95 if category != "general" else 0.05,
                usage=routing_usage if routing_usage is not None else usage(3, 1, 0.001),
                duration_ms=3,
            )
        ),
        advise=AsyncMock(
            return_value=ConsultationAdvice(
                content=ADVICE,
                sources=[
                    ConsultationSource(
                        title="Caffeine evidence", url="https://example.org/evidence"
                    )
                ],
                usage=advice_usage if advice_usage is not None else usage(20, 5, 0.02),
                model="openai/consultant-test",
                reasoning_effort="medium",
                duration_ms=12,
                search_used=True,
            )
        ),
    )


def pipeline(advisor=None, *, vertex=None, alternative=None, enabled=True):
    return ConsultativeGenerationService(
        advisor or consultant(), vertex or Author(), alternative or Author(), enabled=enabled
    )


async def collect(service, req):
    return [chunk async for chunk in service.stream_chat_completion(req)]


@pytest.fixture(autouse=True)
def telemetry(monkeypatch):
    capture = Mock()
    monkeypatch.setattr(consultative_generation, "capture_span", capture)
    return capture


@pytest.mark.parametrize("provider", ["vertex", "openrouter"])
async def test_disabled_pipeline_is_byte_identical_and_does_not_consult(provider):
    req = request(provider)
    expected = [": provider heartbeat\n\n", *successful_events(req, "Unchanged answer")]
    author = Author({"response": expected})
    other = Author()
    advisor = consultant()
    service = pipeline(
        advisor,
        vertex=author if provider == "vertex" else other,
        alternative=author if provider == "openrouter" else other,
        enabled=False,
    )
    assert await collect(service, req) == expected
    advisor.classify.assert_not_awaited()
    advisor.advise.assert_not_awaited()
    assert not other.requests


async def test_general_turn_uses_original_author_once_and_accounts_for_routing():
    advisor = consultant("general")
    author = Author()
    events = await collect(pipeline(advisor, vertex=author), request())
    assert visible_text(events) == FINAL
    assert [req.generation_stage for req in author.requests] == ["response"]
    advisor.advise.assert_not_awaited()
    reported = final_usage(events)
    assert reported["total_tokens"] == 16
    assert reported["cost"] == pytest.approx(0.011)
    assert reported["consultation"]["status"] == "skipped"
    assert events.count(DONE) == 1


@pytest.mark.parametrize("provider", ["vertex", "openrouter"])
@pytest.mark.parametrize("category", ["general", "medical"])
async def test_real_serialized_chunks_with_null_usage_preserve_successful_public_stream(
    provider, category
):
    req = request(provider)

    def serialized_events(text):
        chunks = [
            ChatCompletionChunk(
                id="chatcmpl-real-serializer",
                created=1,
                model=req.model,
                choices=[{"index": 0, "delta": {"role": "assistant"}}],
            ),
            ChatCompletionChunk(
                id="chatcmpl-real-serializer",
                created=1,
                model=req.model,
                choices=[{"index": 0, "delta": {"content": text}}],
            ),
            ChatCompletionChunk(
                id="chatcmpl-real-serializer",
                created=1,
                model=req.model,
                choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
                usage=usage(),
            ),
        ]
        # GenerationService serializes with defaults, including usage:null on role/content.
        assert chunks[0].model_dump()["usage"] is None
        assert chunks[1].model_dump()["usage"] is None
        return [*[f"data: {chunk.model_dump_json()}\n\n" for chunk in chunks], DONE]

    author = Author(
        {
            "response": serialized_events(FINAL),
            "draft": serialized_events(DRAFT),
            "final": serialized_events(FINAL),
        }
    )
    service = pipeline(
        consultant(category),
        vertex=author if provider == "vertex" else Author(),
        alternative=author if provider == "openrouter" else Author(),
    )
    events = await collect(service, req)
    assert visible_text(events) == FINAL
    assert DRAFT not in "".join(events)
    assert ADVICE not in "".join(events)
    assert not any(chunk.get("error") for chunk in data_chunks(events))
    assert events.count(DONE) == 1
    report = final_usage(events)
    assert report["total_tokens"] == (53 if category == "medical" else 16)
    assert report["consultation"]["status"] == ("consulted" if category == "medical" else "skipped")


@pytest.mark.parametrize("provider", ["vertex", "openrouter"])
@pytest.mark.parametrize("category", ["medical", "mixed"])
async def test_context_and_images_reach_consultant_but_only_author_final_is_published(
    provider, category
):
    req = request(provider)
    original = req.model_copy(deep=True)
    author, other = Author(), Author()
    advisor = consultant(category)
    service = pipeline(
        advisor,
        vertex=author if provider == "vertex" else other,
        alternative=author if provider == "openrouter" else other,
    )
    events = await collect(service, req)
    assert visible_text(events) == FINAL
    assert DRAFT not in "".join(events)
    assert ADVICE not in "".join(events)
    assert req == original
    assert [r.generation_stage for r in author.requests] == ["draft", "final"]
    draft_req, final_req = author.requests
    assert draft_req.messages == original.messages
    assert draft_req.system_instruction == original.system_instruction
    assert draft_req.compilation == final_req.compilation == original.compilation
    assert final_req.messages[: len(original.messages)] == original.messages
    assert DRAFT in json.dumps(final_req.model_dump())
    assert ADVICE in json.dumps(final_req.model_dump())
    advice_req, advice_draft, _ = advisor.advise.await_args.args
    assert advice_req == original
    assert advice_draft == DRAFT
    assert not other.requests
    chunks = data_chunks(events)
    assert all(chunk["model"] == original.model for chunk in chunks)
    assert all(chunk["provider"] == original.provider for chunk in chunks)
    assert events.count(DONE) == 1
    report = final_usage(events)
    assert report["consultation"]["status"] == "consulted"
    assert report["consultation"]["model"] == "openai/consultant-test"


async def test_usage_sums_private_work_but_grounding_and_cache_describe_published_answer():
    req = request()
    draft_sources = [{"title": "Draft source", "uri": "https://example.org/draft"}]
    final_sources = [{"title": "Final source", "uri": "https://example.org/final"}]
    author = Author(
        {
            "draft": successful_events(
                req,
                DRAFT,
                usage(
                    10,
                    2,
                    0.01,
                    grounding_sources=draft_sources,
                    grounding_enabled=True,
                    cache_used=True,
                    thoughts_tokens=4,
                ),
            ),
            "final": successful_events(
                req,
                FINAL,
                usage(
                    30,
                    6,
                    0.03,
                    grounding_sources=final_sources,
                    grounding_enabled=True,
                    cache_used=False,
                    thoughts_tokens=7,
                ),
            ),
        }
    )
    advisor = consultant()
    events = await collect(pipeline(advisor, vertex=author), req)
    report = final_usage(events)
    assert report["prompt_tokens"] == 63
    assert report["completion_tokens"] == 14
    assert report["total_tokens"] == 77
    assert report["thoughts_tokens"] == 11
    assert report["cost"] == pytest.approx(0.061)
    assert report["grounding_sources"] == final_sources
    assert report["cache_used"] is False
    assert advisor.advise.await_args.args[2] == draft_sources
    assert report["consultation"]["sources"] == [
        {
            "title": "Caffeine evidence",
            "url": "https://example.org/evidence",
            "provenance": "provider_annotation",
        }
    ]
    assert report["consultation"]["cost_complete"] is True


@pytest.mark.parametrize("missing_stage", ["routing", "draft", "advice", "final"])
async def test_unknown_stage_cost_is_not_misreported_as_complete_zero(missing_stage):
    req = request()
    amounts = {
        stage: usage(1, 1, None if stage == missing_stage else 0.01)
        for stage in ("routing", "draft", "advice", "final")
    }
    advisor = consultant(routing_usage=amounts["routing"], advice_usage=amounts["advice"])
    author = Author(
        {
            "draft": successful_events(req, DRAFT, amounts["draft"]),
            "final": successful_events(req, FINAL, amounts["final"]),
        }
    )
    report = final_usage(await collect(pipeline(advisor, vertex=author), req))
    assert report["cost"] is None
    assert report["total_tokens"] == 8
    assert report["consultation"]["cost_complete"] is False
    assert report["consultation"]["known_cost_usd"] == pytest.approx(0.03)


async def test_byok_upstream_cost_and_reasoning_tokens_are_accounted_for():
    advisor = consultant(
        routing_usage={
            "input_tokens": 3,
            "output_tokens": 1,
            "cost": 0,
            "is_byok": True,
            "cost_details": {"upstream_inference_cost": 0.002},
        },
        advice_usage=usage(
            20,
            5,
            0,
            is_byok=True,
            cost_details={"upstream_inference_cost": 0.04},
            completion_tokens_details={"reasoning_tokens": 3},
        ),
    )
    report = final_usage(await collect(pipeline(advisor), request()))
    assert report["cost"] == pytest.approx(0.062)
    assert report["thoughts_tokens"] == 3
    assert report["total_tokens"] == 53


@pytest.mark.parametrize("failure_stage", ["routing", "advice"])
@pytest.mark.parametrize("failure_type", ["exception", "timeout"])
async def test_consultant_errors_and_hard_timeouts_fail_open(failure_stage, failure_type):
    advisor = consultant()
    cancelled = asyncio.Event()

    async def unavailable(*args):
        if failure_type == "exception":
            raise RuntimeError("provider unavailable")
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    if failure_stage == "routing":
        advisor.classify.side_effect = unavailable
        advisor.routing_timeout_seconds = 0.01
    else:
        advisor.advise.side_effect = unavailable
        advisor.timeout_seconds = 0.01
    author = Author()
    events = await asyncio.wait_for(collect(pipeline(advisor, vertex=author), request()), 1)
    assert visible_text(events) == (FINAL if failure_stage == "routing" else DRAFT)
    assert ADVICE not in "".join(events)
    assert events.count(DONE) == 1
    report = final_usage(events)
    assert report["consultation"]["status"] == f"{failure_stage}_failed"
    assert report["total_tokens"] == (12 if failure_stage == "routing" else 16)
    if failure_type == "timeout":
        assert cancelled.is_set(), "Timeout must cancel the upstream consultation"
    if failure_stage == "routing":
        advisor.advise.assert_not_awaited()


@pytest.mark.parametrize("failure", ["error", "exception", "empty", "usage_only"])
async def test_final_failure_before_content_replays_draft_once_without_private_advice(failure):
    req = request()
    scripts = {
        "error": [event(req, error="upstream unavailable")],
        "exception": [RuntimeError("upstream unavailable")],
        "empty": [DONE],
        "usage_only": [event(req, usage=usage(1, 1, 0.001), finish="stop"), DONE],
    }
    events = await collect(pipeline(vertex=Author({"final": scripts[failure]})), req)
    assert visible_text(events) == DRAFT
    assert ADVICE not in "".join(events)
    assert not any(chunk.get("error") for chunk in data_chunks(events))
    assert events.count(DONE) == 1
    report = final_usage(events)
    assert report["consultation"]["status"] == "final_failed"
    assert report["total_tokens"] == (43 if failure == "usage_only" else 41)


@pytest.mark.parametrize("failure", ["error", "exception", "eof"])
@pytest.mark.parametrize("provider", ["vertex", "openrouter"])
async def test_failure_after_final_text_never_concatenates_draft_or_leaves_stream_unterminated(
    failure, provider
):
    req = request(provider)
    tail = [event(req, error="upstream disconnected")]
    if provider == "vertex":
        tail.append(DONE)
    if failure == "exception":
        tail = [RuntimeError("provider secret must not leak")]
    elif failure == "eof":
        tail = []
    author = Author({"final": [event(req, text="Published partial"), *tail]})
    service = pipeline(
        vertex=author if provider == "vertex" else Author(),
        alternative=author if provider == "openrouter" else Author(),
    )
    events = await collect(service, req)
    assert visible_text(events) == "Published partial"
    assert DRAFT not in "".join(events)
    assert ADVICE not in "".join(events)
    assert "provider secret must not leak" not in "".join(events)
    assert any(chunk.get("error") for chunk in data_chunks(events))
    assert events.count(DONE) == 1
    report = final_usage(events)
    assert report["total_tokens"] == 41
    assert report["prompt_tokens"] == 33
    assert report["completion_tokens"] == 8
    assert report["cost"] is None
    summary = report["consultation"]
    assert summary["status"] == "final_failed_partial"
    assert summary["cost_complete"] is False
    assert summary["known_cost_usd"] == pytest.approx(0.031)
    assert "published" not in summary["stages"], "Missing author usage must remain unknown"
    usage_position = next(i for i, frame in enumerate(events) if '"usage"' in frame)
    assert usage_position < events.index(DONE)


@pytest.mark.parametrize("failure", ["error", "exception", "eof"])
async def test_usage_before_final_failure_is_reported_once_with_terminal_failure_status(failure):
    req = request()
    tail = {
        "error": [event(req, error="provider disconnected"), DONE],
        "exception": [RuntimeError("provider disconnected")],
        "eof": [],
    }[failure]
    author = Author(
        {
            "final": [
                event(req, text="Published partial"),
                event(req, usage=usage(), finish="stop"),
                *tail,
            ]
        }
    )
    events = await collect(pipeline(vertex=author), req)
    assert visible_text(events) == "Published partial"
    assert any(chunk.get("error") for chunk in data_chunks(events))
    report = final_usage(events)
    assert report["total_tokens"] == 53
    assert report["consultation"]["status"] == "final_failed_partial"
    assert report["consultation"]["cost_complete"] is False
    assert report["consultation"]["known_cost_usd"] == pytest.approx(0.041)
    assert report["cost"] is None
    assert events.count(DONE) == 1


@pytest.mark.parametrize("provider", ["vertex", "openrouter"])
async def test_partial_error_frame_counts_published_usage_once(provider):
    req = request(provider)
    partial_usage = usage(5, 2, 0.004, thoughts_tokens=1)
    author = Author(
        {
            "final": [
                event(req, text="Published partial"),
                event(req, error="upstream disconnected", usage=partial_usage),
                *([DONE] if provider == "vertex" else []),
            ]
        }
    )
    service = pipeline(
        vertex=author if provider == "vertex" else Author(),
        alternative=author if provider == "openrouter" else Author(),
    )
    events = await collect(service, req)
    assert visible_text(events) == "Published partial"
    assert DRAFT not in "".join(events)
    report = final_usage(events)
    assert report["prompt_tokens"] == 38
    assert report["completion_tokens"] == 10
    assert report["total_tokens"] == 48
    assert report["thoughts_tokens"] == 1
    assert report["consultation"]["stages"]["published"]["total_tokens"] == 7
    assert report["consultation"]["known_cost_usd"] == pytest.approx(0.035)
    assert report["consultation"]["cost_complete"] is False
    assert report["consultation"]["status"] == "final_failed_partial"
    assert events.count(DONE) == 1


@pytest.mark.parametrize("failure_stage", ["routing", "advice"])
async def test_consultation_error_preserves_known_billable_usage_and_marks_incomplete(
    failure_stage,
):
    advisor = consultant()
    error = ConsultationError("invalid_response")
    error.usage = usage(7, 3, 0.005)
    if failure_stage == "routing":
        advisor.classify.side_effect = error
    else:
        advisor.advise.side_effect = error
    events = await collect(pipeline(advisor), request())
    assert visible_text(events) == (FINAL if failure_stage == "routing" else DRAFT)
    report = final_usage(events)
    assert report["total_tokens"] == (22 if failure_stage == "routing" else 26)
    assert report["cost"] is None
    summary = report["consultation"]
    assert summary["status"] == f"{failure_stage}_failed"
    assert summary["cost_complete"] is False
    assert summary["known_cost_usd"] == pytest.approx(
        0.015 if failure_stage == "routing" else 0.016
    )
    assert events.count(DONE) == 1


@pytest.mark.parametrize("provider", ["vertex", "openrouter"])
async def test_legacy_system_message_persona_survives_final_author_pass(provider):
    req = request(provider)
    req.system_instruction = None
    legacy_persona = "Legacy persona: speak gently in Spanish and remember my name is Leidy."
    req.messages.insert(0, ChatMessage(role="system", content=legacy_persona))
    original = req.model_copy(deep=True)
    author = Author()
    service = pipeline(
        vertex=author if provider == "vertex" else Author(),
        alternative=author if provider == "openrouter" else Author(),
    )
    events = await collect(service, req)
    assert visible_text(events) == FINAL
    final_req = author.requests[-1]
    trusted_persona = (
        (final_req.system_instruction or "")
        + "\n"
        + "\n".join(
            str(message.content) for message in final_req.messages if message.role == "system"
        )
    )
    assert legacy_persona in trusted_persona
    assert req == original


async def test_draft_failure_is_not_sent_to_consultant():
    req = request()
    author = Author(
        {"draft": [event(req, text="Partial draft"), event(req, error="provider error"), DONE]}
    )
    advisor = consultant()
    events = await collect(pipeline(advisor, vertex=author), req)
    assert visible_text(events) == "Partial draft"
    assert any(chunk.get("error") for chunk in data_chunks(events))
    advisor.advise.assert_not_awaited()
    assert [r.generation_stage for r in author.requests] == ["draft"]


async def test_failed_draft_without_author_usage_preserves_routing_bill_and_original_error():
    req = request()
    original_error = "Author provider could not connect"
    author = Author({"draft": [event(req, error=original_error), DONE]})
    advisor = consultant()
    events = await collect(pipeline(advisor, vertex=author), req)
    chunks = data_chunks(events)
    assert visible_text(events) == ""
    errors = [chunk["error"] for chunk in chunks if chunk.get("error")]
    assert errors == [{"message": original_error}]
    assert all(chunk["model"] == req.model for chunk in chunks)
    report = final_usage(events)
    assert report["prompt_tokens"] == 3
    assert report["completion_tokens"] == 1
    assert report["total_tokens"] == 4
    assert report["cost"] is None
    summary = report["consultation"]
    assert summary["status"] == "draft_failed"
    assert summary["known_cost_usd"] == pytest.approx(0.001)
    assert summary["cost_complete"] is False
    assert "published" not in summary["stages"]
    assert events.count(DONE) == 1
    assert events[-1] == DONE
    advisor.advise.assert_not_awaited()
    assert [r.generation_stage for r in author.requests] == ["draft"]


async def test_analytics_failure_never_interrupts_chat(telemetry):
    telemetry.side_effect = RuntimeError("analytics unavailable")
    events = await collect(pipeline(), request())
    assert visible_text(events) == FINAL
    assert events.count(DONE) == 1
    telemetry.assert_called_once()


async def test_pipeline_emits_traceable_metadata_without_draft_in_pipeline_analytics(telemetry):
    events = await collect(pipeline(), request())
    telemetry.assert_called_once()
    args, kwargs = telemetry.call_args
    assert args == ("test-owner", "test-trace")
    assert kwargs["name"] == "consultation.pipeline"
    props = kwargs["properties"]
    assert props["$ai_session_id"] == "test-session"
    assert props["consultation_status"] == "consulted"
    assert props["consultation_source_count"] == 1
    assert props["consultation_search_used"] is True
    assert props["summary"]["stages"] == final_usage(events)["consultation"]["stages"]
    assert DRAFT not in json.dumps(props)
    assert ADVICE not in json.dumps(props)


async def test_closing_client_stream_cancels_consultant_worker():
    advisor = consultant()
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def blocked(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    advisor.advise.side_effect = blocked
    service = pipeline(advisor)
    stream = service.stream_chat_completion(request())
    next_event = asyncio.create_task(anext(stream))
    await asyncio.wait_for(entered.wait(), 1)
    next_event.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_event
    await stream.aclose()
    await asyncio.wait_for(cancelled.wait(), 1)


async def test_closing_client_stream_closes_active_author_generator():
    entered, closed = asyncio.Event(), asyncio.Event()

    class BlockingAuthor:
        async def stream_chat_completion(self, req, *, cache_manager=None):
            try:
                entered.set()
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed.set()

    stream = pipeline(vertex=BlockingAuthor()).stream_chat_completion(request())
    next_event = asyncio.create_task(anext(stream))
    await asyncio.wait_for(entered.wait(), 1)
    next_event.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_event
    await stream.aclose()
    await asyncio.wait_for(closed.wait(), 1)


async def test_disconnect_with_full_worker_queue_closes_final_author_and_records_cancelled(
    monkeypatch, telemetry
):
    queue_blocked = asyncio.Event()
    final_closed = asyncio.Event()
    real_queue = asyncio.Queue

    class ObservedQueue(real_queue):
        async def put(self, item):
            if self.full():
                queue_blocked.set()
            await super().put(item)

    class ManyChunksAuthor:
        async def stream_chat_completion(self, req, *, cache_manager=None):
            if req.generation_stage != "final":
                for item in successful_events(req, DRAFT):
                    yield item
                return
            try:
                for _ in range(100):
                    yield event(req, text="Final author fragment ")
                yield event(req, usage=usage(), finish="stop")
                yield DONE
            finally:
                final_closed.set()

    monkeypatch.setattr(consultative_generation.asyncio, "Queue", ObservedQueue)
    stream = pipeline(vertex=ManyChunksAuthor()).stream_chat_completion(request())
    first = await anext(stream)
    assert visible_text([first]) == "Final author fragment "
    await asyncio.wait_for(queue_blocked.wait(), 1)
    assert not final_closed.is_set()
    await asyncio.wait_for(stream.aclose(), 1)
    assert final_closed.is_set()
    telemetry.assert_called_once()
    assert telemetry.call_args.kwargs["properties"]["consultation_status"] == "cancelled"


async def test_slow_consultation_heartbeats_are_sse_comments_without_visible_content(monkeypatch):
    advisor = consultant()
    release = asyncio.Event()
    normal_classify = advisor.classify.return_value

    async def delayed(*args):
        await release.wait()
        return normal_classify

    advisor.classify.side_effect = delayed
    real_wait_for = asyncio.wait_for

    async def short_heartbeat(awaitable, timeout):
        return await real_wait_for(awaitable, 0.01 if timeout == 10 else timeout)

    monkeypatch.setattr(consultative_generation.asyncio, "wait_for", short_heartbeat)
    stream = pipeline(advisor).stream_chat_completion(request())
    heartbeat = await anext(stream)
    assert heartbeat.startswith(":")
    assert not data_chunks([heartbeat])
    assert visible_text([heartbeat]) == ""
    release.set()
    events = [heartbeat, *[entry async for entry in stream]]
    assert visible_text(events) == FINAL
    assert events.count(DONE) == 1


@pytest.mark.parametrize("provider", ["vertex", "openrouter"])
async def test_production_router_refreshes_memory_before_rag_routing_and_selected_author(
    monkeypatch, provider
):
    app = FastAPI()
    app.include_router(router)
    author, other = Author(), Author()
    advisor = consultant()
    service = pipeline(
        advisor,
        vertex=author if provider == "vertex" else other,
        alternative=author if provider == "openrouter" else other,
    )
    app.state.generation_service = service.generation
    app.state.openrouter_generation_service = service.openrouter
    app.state.consultative_generation_service = service
    app.state.cache_manager = object()
    app.state.graphiti = object()

    async def refresh_compilation(req):
        req.compilation = "Current server-side memory: caffeine sensitivity."

    app.state.hydration_service = SimpleNamespace(
        refresh_chat_compilation=AsyncMock(side_effect=refresh_compilation)
    )
    monkeypatch.setattr(
        security, "get_settings", lambda: SimpleNamespace(synapse_api_secret="test-secret")
    )

    async def rag(req, graph):
        assert req.compilation == "Current server-side memory: caffeine sensitivity."
        req.system_instruction += " RAG: user recently reported headaches."
        return GraphRagOutcome(enabled=False, skip_reason="test")

    monkeypatch.setattr("app.api.routes.maybe_run_graph_rag", rag)
    monkeypatch.setattr("app.api.routes.capture_trace", Mock())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=request(provider).model_dump(mode="json"),
            headers={"X-API-SECRET": "test-secret"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert visible_text([response.text]) == FINAL
    assert DRAFT not in response.text
    assert ADVICE not in response.text
    route_req = advisor.classify.await_args.args[0]
    assert "RAG: user recently reported headaches" in route_req.system_instruction
    assert route_req.compilation == "Current server-side memory: caffeine sensitivity."
    assert all(r.compilation == route_req.compilation for r in author.requests)
    assert route_req.posthog_trace_id
    assert [r.generation_stage for r in author.requests] == ["draft", "final"]
    assert not other.requests
    if provider == "openrouter":
        author.validate_request.assert_called_once()
        assert all(r.cache_name is None for r in author.requests)
    else:
        assert author.cache_managers == [app.state.cache_manager, app.state.cache_manager]
    assert final_usage([response.text])["consultation"]["trace_id"] == route_req.posthog_trace_id
