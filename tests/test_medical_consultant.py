import asyncio
import hashlib
import json

import httpx
import pytest

from app.schemas.models import ChatCompletionRequest, ChatMessage
from app.services.medical_consultant import ConsultationError, MedicalConsultantClient


def request(**kwargs):
    return ChatCompletionRequest(
        messages=kwargs.pop("messages", [ChatMessage(role="user", content="¿Este suplemento?")]),
        user_id="demo-user",
        session_id="demo-session",
        posthog_trace_id="trace-consultation",
        **kwargs,
    )


def make_client(response, calls, **kwargs):
    def handle(req):
        calls.append(json.loads(req.content))
        if isinstance(response, Exception):
            raise response
        return response

    return MedicalConsultantClient(
        "private-api-key", client=httpx.AsyncClient(transport=httpx.MockTransport(handle)), **kwargs
    )


def decision_response(category="medical", **kwargs):
    return httpx.Response(
        200,
        json={
            "answers": {
                "consultation_route": {
                    "type": "choice",
                    "choice": category,
                    "probabilities": {category: 0.94},
                }
            },
            "usage": {"input_tokens": 44, "output_tokens": 0, "cost": 0.0001},
            **kwargs,
        },
    )


def advice_response(**message):
    return httpx.Response(
        200,
        json={
            "model": "openai/gpt-6.1-sol",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "Considera matizar la garantía.", **message},
                }
            ],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 90,
                "cost": 0.02,
                "completion_tokens_details": {"reasoning_tokens": 70},
                "server_tool_use_details": {"web_search_requests": 1},
            },
        },
    )


async def test_routing_followup_preserves_recent_discussion_and_current_image(monkeypatch):
    events = []
    monkeypatch.setattr(
        "app.services.medical_consultant.capture_generation",
        lambda *args, **kwargs: events.append((args, kwargs)),
    )
    calls = []
    consultant = make_client(decision_response(), calls)
    req = request(
        compilation="Private medical memory must not become a blanket trigger.",
        messages=[
            ChatMessage(role="system", content="Old persona medical text"),
            ChatMessage(role="user", content="¿Agrego cafeína para entrenar?"),
            ChatMessage(role="assistant", content="Miremos la dosis y teanina."),
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "¿Este?"},
                    {"type": "image_url", "image_url": {"url": "https://example.org/label.jpg"}},
                ],
            ),
        ],
    )
    result = await consultant.classify(req)
    state = calls[0]["state"]
    assert result.category == "medical"
    assert "cafeína" in state[0] and "¿Este?" in state[0]
    assert state[1]["image_url"]["url"] == "https://example.org/label.jpg"
    assert "Private medical memory" not in state[0]
    assert "Old persona" not in state[0]
    assert isinstance(calls[0]["questions"], dict)
    assert isinstance(calls[0]["questions"]["consultation_route"]["criteria"], dict)
    assert result.usage["prompt_tokens"] == 44
    assert result.usage["cost"] == 0.0001
    assert events[0][1]["session_id"] == "demo-session"
    assert events[0][0] == ("demo-user", "trace-consultation")
    assert events[0][1]["properties"]["consultation_stage"] == "routing"


async def test_psychological_routing_criteria_exclude_trauma_and_topic_changes():
    calls = []
    consultant = make_client(decision_response("general"), calls)
    req = request(
        messages=[
            ChatMessage(role="user", content="¿Cafeína?"),
            ChatMessage(role="assistant", content="Puede tener efectos secundarios."),
            ChatMessage(role="user", content="Gracias. Quiero hablar del trauma con mi familia."),
        ]
    )
    result = await consultant.classify(req)
    question = calls[0]["questions"]["consultation_route"]
    assert result.category == "general"
    assert "trauma" in question["criteria"]["general"]
    assert "farewell or topic change" in question["instructions"]
    assert "immediately active" in question["instructions"]


async def test_routing_context_is_bounded_and_full_current_text_has_priority():
    calls = []
    consultant = make_client(decision_response(), calls)
    req = request(
        messages=[
            *[ChatMessage(role="user", content=f"old-{index}-" + "x" * 8000) for index in range(9)],
            ChatMessage(role="user", content="y" * 6000 + "CURRENT-SYMPTOM"),
        ]
    )
    await consultant.classify(req)
    state = json.loads(calls[0]["state"])["recent_conversation"]
    assert len(state) == 6
    assert len(state[-1]["text"]) == 4000
    assert state[-1]["text"].endswith("CURRENT-SYMPTOM")
    assert all(len(record["text"]) <= 1200 for record in state[:-1])


async def test_advisor_keeps_complete_context_images_and_persona_only_as_quoted_data(monkeypatch):
    events = []
    monkeypatch.setattr(
        "app.services.medical_consultant.capture_generation",
        lambda *args, **kwargs: events.append(kwargs),
    )
    calls = []
    consultant = make_client(
        advice_response(
            annotations=[
                {
                    "type": "url_citation",
                    "url_citation": {"title": "Official", "url": "https://gov.test/a"},
                },
                {
                    "type": "url_citation",
                    "url_citation": {"title": "Duplicate", "url": "https://gov.test/a"},
                },
                {"type": "url_citation", "url_citation": {"url": "javascript:alert(1)"}},
            ]
        ),
        calls,
    )
    memory = "known-personal-history " * 2000 + "MEMORY-END"
    req = request(
        system_instruction="You are Leidy's companion, adopt this persona.",
        compilation=memory,
        messages=[
            ChatMessage(role="system", content="legacy persona"),
            ChatMessage(role="assistant", content="HISTORY-START" + "z" * 10000),
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "¿Este?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            ),
        ],
    )
    result = await consultant.advise(
        req, "BORRADOR", [{"title": "Prior", "uri": "https://prior.test"}]
    )
    payload = calls[0]
    assert payload["messages"][0]["role"] == "system"
    assert "adopt this persona" not in payload["messages"][0]["content"]
    context = json.loads(payload["messages"][1]["content"][0]["text"])
    assert context["personal_memory_and_retrieved_context"] == memory
    assert context["author_system_instruction"] == req.system_instruction
    assert context["conversation"][0]["content"] == "legacy persona"
    assert len(context["conversation"][1]["content"]) > 10000
    assert context["author_draft"] == "BORRADOR"
    assert context["draft_sources_not_independently_verified"][0]["uri"] == "https://prior.test"
    assert payload["messages"][1]["content"][-1]["image_url"]["url"].endswith("AAAA")
    assert payload["transforms"] == []
    assert payload["tools"][0]["type"] == "openrouter:web_search"
    assert payload["provider"]["require_parameters"] is True
    assert payload["max_tool_calls"] == 3
    assert result.sources[0].url == "https://gov.test/a" and len(result.sources) == 1
    assert result.search_used
    assert result.usage["completion_tokens_details"]["reasoning_tokens"] == 70
    assert events[0]["properties"]["$ai_total_cost_usd"] == 0.02
    assert events[0]["properties"]["consultation_policy_version"] == "v2"


async def test_search_can_be_disabled_without_silent_retry():
    calls = []
    consultant = make_client(advice_response(), calls, search_enabled=False, reasoning_effort="low")
    result = await consultant.advise(request(), "draft")
    assert "tools" not in calls[0]
    assert calls[0]["reasoning"]["effort"] == "low"
    assert result.reasoning_effort == "low"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response,category",
    [
        (httpx.Response(401, json={"error": "private-api-key leaked response"}), "upstream_http"),
        (httpx.Response(429, text="sensitive medical text"), "rate_limited"),
        (
            httpx.Response(200, json={"answers": {"consultation_route": {"type": "refusal"}}}),
            "routing_unanswered",
        ),
        (httpx.Response(200, text="not-json private-api-key"), "invalid_response"),
        (httpx.ReadTimeout("private-api-key"), "timeout"),
    ],
)
async def test_failures_raise_sanitized_errors(response, category, caplog):
    calls = []
    consultant = make_client(response, calls)
    with pytest.raises(ConsultationError) as failure:
        await consultant.classify(request())
    assert failure.value.category == category
    assert "private-api-key" not in str(failure.value)
    assert "private-api-key" not in caplog.text
    assert "sensitive medical text" not in caplog.text


@pytest.mark.parametrize(
    "choice,category",
    [
        ({"finish_reason": "length", "message": {"content": "partial"}}, "incomplete_response"),
        ({"finish_reason": "stop", "message": {"content": ""}}, "empty_response"),
        ({"finish_reason": "stop", "message": {"refusal": "details"}}, "refusal"),
        ({"finish_reason": "tool_calls", "message": {"tool_calls": [{}]}}, "incomplete_response"),
    ],
)
async def test_advisor_rejects_unfinished_or_refused_provider_response(choice, category):
    calls = []
    consultant = make_client(httpx.Response(200, json={"choices": [choice]}), calls)
    with pytest.raises(ConsultationError) as failure:
        await consultant.advise(request(), "draft")
    assert failure.value.category == category
    assert len(calls) == 1


async def test_telemetry_failure_does_not_interrupt_consultation(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("private diagnostics")

    monkeypatch.setattr("app.services.medical_consultant.capture_generation", broken)
    consultant = make_client(advice_response(), [])
    result = await consultant.advise(request(), "draft")
    assert result.content


async def test_telemetry_cost_normalization_failure_is_also_nonfatal(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("private diagnostics")

    monkeypatch.setattr("app.services.medical_consultant.openrouter_usage_properties", broken)
    consultant = make_client(decision_response(), [])
    assert (await consultant.classify(request())).category == "medical"


@pytest.mark.parametrize("search_requests,expected_search_used", [(0, False), (2, True)])
async def test_model_links_do_not_claim_search_without_provider_evidence(
    search_requests, expected_search_used
):
    data = advice_response(
        content="Matiza la garantía: [FDA](https://www.fda.gov/caffeine)."
    ).json()
    data["usage"]["server_tool_use_details"]["web_search_requests"] = search_requests
    consultant = make_client(httpx.Response(200, json=data), [])
    result = await consultant.advise(request(), "draft")
    assert result.search_used is expected_search_used
    assert result.search_requests == search_requests
    assert result.sources[0].provenance == "model_link"
    assert result.sources[0].url == "https://www.fda.gov/caffeine"


async def test_provider_annotations_win_over_model_links_without_inventing_provenance():
    data = advice_response(
        content="[Model title](https://gov.test/a) y [Other](https://pubmed.test/study).",
        annotations=[
            {
                "type": "url_citation",
                "url_citation": {"title": "Provider title", "url": "https://gov.test/a"},
            }
        ],
    ).json()
    data["usage"]["server_tool_use_details"]["web_search_requests"] = 0
    consultant = make_client(httpx.Response(200, json=data), [])
    result = await consultant.advise(request(), "draft")
    assert result.search_used
    assert result.search_requests == 0
    assert [source.provenance for source in result.sources] == ["provider_annotation", "model_link"]
    assert result.sources[0].title == "Provider title"


async def test_analytics_input_is_bounded_omits_memory_and_image_data_but_model_input_is_full(
    monkeypatch,
):
    events = []
    monkeypatch.setattr(
        "app.services.medical_consultant.capture_generation",
        lambda *args, **kwargs: events.append(kwargs),
    )
    calls = []
    consultant = make_client(advice_response(), calls)
    req = request(
        compilation="UNIQUE-MEMORY-CONTENT" * 10000,
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "¿Este suplemento?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,PRIVATE_IMAGE_BYTES"},
                    },
                ],
            )
        ],
    )
    await consultant.advise(req, "draft")
    event = events[0]
    assert len(event["input_messages"]) <= 32000
    assert "UNIQUE-MEMORY-CONTENT" not in event["input_messages"]
    assert "PRIVATE_IMAGE_BYTES" not in event["input_messages"]
    assert "data:image" not in event["input_messages"]
    assert event["output"] == "Considera matizar la garantía."
    full_input = json.dumps(calls[0]["messages"], ensure_ascii=False, sort_keys=True)
    assert "UNIQUE-MEMORY-CONTENT" in full_input and "PRIVATE_IMAGE_BYTES" in full_input
    assert event["properties"]["input_chars"] == len(full_input)
    assert event["properties"]["compilation_chars"] == len(req.compilation)
    assert (
        event["properties"]["context_sha256"]
        == hashlib.sha256(full_input.encode("utf-8")).hexdigest()
    )


@pytest.mark.parametrize("stage", ["routing", "advice"])
async def test_hard_deadline_bounds_a_stalled_http_client_and_captures_timeout(stage, monkeypatch):
    events = []
    monkeypatch.setattr(
        "app.services.medical_consultant.capture_generation",
        lambda *args, **kwargs: events.append(kwargs),
    )

    async def stalled(req):
        await asyncio.Event().wait()

    consultant = MedicalConsultantClient(
        "key",
        timeout_seconds=0.005,
        routing_timeout_seconds=0.005,
        client=httpx.AsyncClient(transport=httpx.MockTransport(stalled)),
    )
    with pytest.raises(ConsultationError) as failure:
        if stage == "routing":
            await consultant.classify(request())
        else:
            await consultant.advise(request(), "draft")
    assert failure.value.category == "timeout"
    assert events[0]["properties"]["consultation_failure_category"] == "timeout"


@pytest.mark.parametrize("stage", ["routing", "advice"])
async def test_parent_cancellation_is_recorded_and_reraised(stage, monkeypatch):
    events = []
    monkeypatch.setattr(
        "app.services.medical_consultant.capture_generation",
        lambda *args, **kwargs: events.append(kwargs),
    )
    entered = asyncio.Event()

    async def stalled(req):
        entered.set()
        await asyncio.Event().wait()

    consultant = MedicalConsultantClient(
        "key", client=httpx.AsyncClient(transport=httpx.MockTransport(stalled))
    )
    call = (
        consultant.classify(request())
        if stage == "routing"
        else consultant.advise(request(), "draft")
    )
    task = asyncio.create_task(call)
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events[0]["properties"]["consultation_failure_category"] == "cancelled"
    assert events[0]["properties"]["consultation_stage"] == stage
    assert "cancelled" in events[0]["error"]


async def test_failed_advice_preserves_billed_usage_without_exposing_unpublished_content():
    data = advice_response(content="PRIVATE_UNFINISHED_ADVICE").json()
    data["choices"][0]["finish_reason"] = "length"
    consultant = make_client(httpx.Response(200, json=data), [])
    with pytest.raises(ConsultationError) as failure:
        await consultant.advise(request(), "draft")
    assert failure.value.category == "incomplete_response"
    assert failure.value.usage["prompt_tokens"] == 30
    assert failure.value.usage["completion_tokens"] == 90
    assert failure.value.usage["cost"] == 0.02
    assert failure.value.usage["completion_tokens_details"]["reasoning_tokens"] == 70
    assert "PRIVATE_UNFINISHED_ADVICE" not in str(failure.value)
    assert "PRIVATE_UNFINISHED_ADVICE" not in json.dumps(failure.value.usage)


async def test_invalid_route_preserves_charged_usage():
    consultant = make_client(decision_response("unsupported_route"), [])
    with pytest.raises(ConsultationError) as failure:
        await consultant.classify(request())
    assert failure.value.category == "invalid_route"
    assert failure.value.usage["prompt_tokens"] == 44
    assert failure.value.usage["completion_tokens"] == 0
    assert failure.value.usage["cost"] == 0.0001


async def test_network_failure_has_unknown_empty_usage_and_sanitized_error():
    consultant = make_client(httpx.ConnectError("private-api-key"), [])
    with pytest.raises(ConsultationError) as failure:
        await consultant.advise(request(), "draft")
    assert failure.value.category == "network"
    assert failure.value.usage == {}
    assert "private-api-key" not in str(failure.value)
