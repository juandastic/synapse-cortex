import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import HTTPException

from app.schemas.models import ChatCompletionRequest
from app.services.openrouter_generation import MODEL_CONFIG, OpenRouterGenerationService


def request(**overrides):
    return ChatCompletionRequest(
        **{
            "provider": "openrouter",
            "model": "openai/gpt-6.1-sol",
            "system_instruction": "Persona. Recalled memory: likes hiking.",
            "compilation": "Compiled memory: lives in Bogota.",
            "cache_name": "cachedContents/gemini-only",
            "messages": [
                {"role": "user", "content": "Remember my city?"},
                {"role": "assistant", "content": "Bogota."},
                {"role": "user", "content": "And my hobby?"},
            ],
            **overrides,
        }
    )


class OpenRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = OpenRouterGenerationService("test-secret")

    async def asyncTearDown(self):
        await self.service.close()

    async def test_preserves_persona_compilation_retrieved_memories_and_history(self):
        payload = self.service.build_payload(request())
        self.assertIn("likes hiking", payload["messages"][0]["content"])
        self.assertIn("lives in Bogota", payload["messages"][0]["content"])
        self.assertEqual(
            [m["role"] for m in payload["messages"]],
            ["system", "user", "assistant", "user"],
        )
        self.assertNotIn("cache_name", payload)
        self.assertNotIn("cachedContents", json.dumps(payload))
        self.assertNotIn("google_search", json.dumps(payload))

    async def test_all_five_models_use_requested_reasoning(self):
        for model in MODEL_CONFIG:
            reasoning = self.service.build_payload(request(model=model))["reasoning"]
            self.assertTrue(reasoning["exclude"])
            if model in list(MODEL_CONFIG)[:3]:
                self.assertEqual(reasoning["effort"], "high")
            else:
                self.assertTrue(reasoning["enabled"])

    async def test_images_in_earlier_turn_are_never_silently_dropped(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    }
                ],
            },
            {"role": "user", "content": "What was in the image?"},
        ]
        for model in ["deepseek/deepseek-v4-pro-0813", "qwen/qwen3.7-max"]:
            with self.assertRaises(HTTPException) as error:
                self.service.build_payload(request(model=model, messages=messages))
            self.assertEqual(error.exception.status_code, 400)
        payload = self.service.build_payload(request(messages=messages))
        self.assertEqual(payload["messages"][1]["content"], messages[0]["content"])

    async def test_disallows_unknown_models_and_missing_key(self):
        with self.assertRaises(HTTPException):
            self.service.build_payload(request(model="openrouter/auto"))
        self.service._api_key = ""
        with self.assertRaises(HTTPException) as error:
            self.service.build_payload(request())
        self.assertEqual(error.exception.status_code, 503)

    async def _stream(self, events, status=200):
        seen = []

        def handle(upstream):
            seen.append(json.loads(upstream.content))
            return httpx.Response(status, text=events)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            service = OpenRouterGenerationService("test-secret", client)
            req = request(
                user_id="user", session_id="session", posthog_trace_id="trace"
            )
            req.rag_usage_fields = {"rag_enabled": True, "rag_nodes": 2}
            chunks = [chunk async for chunk in service.stream_chat_completion(req)]
        return chunks, seen

    async def test_stream_normalizes_usage_and_never_exposes_reasoning(self):
        chunks, seen = await self._stream(
            ": heartbeat\n\n"
            'data: {"model":"openai/gpt-6.1-sol","choices":[{"delta":{"reasoning":"private thoughts","content":"Hiking"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"cost":0.001,"completion_tokens_details":{"reasoning_tokens":12}}}\n\n'
            "data: [DONE]\n\n"
        )
        self.assertEqual(len(seen), 1)
        self.assertNotIn("private thoughts", "".join(chunks))
        self.assertNotIn("test-secret", "".join(chunks))
        final = json.loads(chunks[-2][6:])
        self.assertEqual(final["usage"]["thoughts_tokens"], 12)
        self.assertEqual(final["usage"]["rag_nodes"], 2)
        self.assertEqual(final["usage"]["cost"], 0.001)
        self.assertFalse(final["usage"]["grounding_enabled"])
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

    async def test_truncated_empty_and_provider_error_streams_are_failures(self):
        for events, status in [
            ('data: {"choices":[{"delta":{"content":"Partial"}}]}\n\n', 200),
            (
                'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n',
                200,
            ),
            ('data: {"error":{"message":"Rate limit"}}\n\n', 200),
            ('{"error":{"message":"Invalid API key"}}', 401),
        ]:
            chunks, seen = await self._stream(events, status)
            self.assertEqual(len(seen), 1)
            self.assertIn('"error"', chunks[-1])
            self.assertNotIn("[DONE]", chunks[-1])

    async def test_posthog_records_generation_usage_cost_and_trace_in_seconds(self):
        analytics = Mock()
        events = (
            'data: {"model":"openai/gpt-6.1-sol","choices":[{"delta":{"reasoning":"private thoughts","content":"Hi "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"there"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"cost":0.001,"completion_tokens_details":{"reasoning_tokens":12},"prompt_tokens_details":{"cached_tokens":50}}}\n\n'
            "data: [DONE]\n\n"
        )
        with (
            patch("app.core.posthog._posthog_client", analytics),
            patch(
                "app.services.openrouter_generation.time.monotonic",
                side_effect=[100, 102],
            ),
        ):
            chunks, _ = await self._stream(events)
        analytics.capture.assert_called_once()
        event = analytics.capture.call_args.kwargs
        self.assertEqual(event["event"], "$ai_generation")
        self.assertEqual(event["distinct_id"], "user")
        props = event["properties"]
        self.assertEqual(props["$ai_trace_id"], "trace")
        self.assertEqual(props["$ai_session_id"], "session")
        self.assertEqual(props["$ai_provider"], "openrouter")
        self.assertEqual(props["$ai_model"], "openai/gpt-6.1-sol")
        self.assertEqual(props["$ai_input_tokens"], 100)
        self.assertEqual(props["$ai_output_tokens"], 20)
        self.assertEqual(props["$ai_total_tokens"], 120)
        self.assertEqual(props["$ai_reasoning_tokens"], 12)
        self.assertEqual(props["$ai_cache_read_input_tokens"], 50)
        self.assertEqual(props["$ai_total_cost_usd"], 0.001)
        self.assertEqual(props["$ai_latency"], 2)
        self.assertEqual(props["rag_nodes"], 2)
        self.assertEqual(props["finish_reason"], "stop")
        self.assertEqual(
            props["$ai_output_choices"], [{"role": "assistant", "content": "Hi there"}]
        )
        self.assertIn("lives in Bogota", props["$ai_input"][0]["content"])
        self.assertNotIn("$ai_is_error", props)
        self.assertNotIn("private thoughts", json.dumps(event))
        self.assertNotIn("test-secret", json.dumps(event))
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

    async def test_posthog_records_http_truncation_and_partial_stream_failures_once(
        self,
    ):
        for events, status, output in [
            ('{"error":{"message":"Rate limited"}}', 429, ""),
            ('data: {"choices":[{"delta":{"content":"Partial"}}]}\n\n', 200, "Partial"),
            (
                (
                    'data: {"model":"openai/gpt-6.1-sol","choices":[{"delta":{"content":"Partial"}}],"usage":{"prompt_tokens":10,"completion_tokens":2,"cost":0.005}}\n\n'
                    'data: {"error":{"message":"Provider disconnected"}}\n\n'
                ),
                200,
                "Partial",
            ),
        ]:
            with self.subTest(status=status, events=events):
                analytics = Mock()
                with patch("app.core.posthog._posthog_client", analytics):
                    chunks, _ = await self._stream(events, status)
                analytics.capture.assert_called_once()
                props = analytics.capture.call_args.kwargs["properties"]
                self.assertTrue(props["$ai_is_error"])
                self.assertTrue(props["$ai_error"])
                self.assertEqual(props["finish_reason"], "error")
                self.assertEqual(props["$ai_output_choices"][0]["content"], output)
                self.assertEqual(props["$ai_trace_id"], "trace")
                self.assertEqual(props["$ai_session_id"], "session")
                self.assertIn('"error"', chunks[-1])
                if "Provider disconnected" in events:
                    self.assertEqual(props["$ai_input_tokens"], 10)
                    self.assertEqual(props["$ai_total_cost_usd"], 0.005)
                else:
                    self.assertNotIn("$ai_total_cost_usd", props)

    async def test_posthog_failure_does_not_interrupt_successful_chat(self):
        analytics = Mock()
        analytics.capture.side_effect = RuntimeError("Analytics unavailable")
        with patch("app.core.posthog._posthog_client", analytics):
            chunks, _ = await self._stream(
                'data: {"choices":[{"delta":{"content":"Answer"}}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            )
        analytics.capture.assert_called_once()
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertNotIn('"error"', "".join(chunks))


class DispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_rag_precedes_openrouter_and_vertex_service_is_not_called(self):
        from app.api.routes import chat_completions
        from app.services.graph_rag import GraphRagOutcome

        calls = []

        class Generator:
            def validate_request(self, req):
                calls.append("validate")

            async def stream_chat_completion(self, req):
                calls.append("openrouter")
                self.assert_request = req
                yield "answer"

        generator = Generator()
        vertex = AsyncMock()

        async def rag(req, graph):
            calls.append("rag")
            req.system_instruction += " Additional retrieved memory."
            return GraphRagOutcome(enabled=False, skip_reason="test")

        req = request()
        with patch("app.api.routes.maybe_run_graph_rag", side_effect=rag):
            response = await chat_completions(
                req, "secret", vertex, generator, object(), object()
            )
            self.assertEqual(
                [chunk async for chunk in response.body_iterator], ["answer"]
            )
        self.assertEqual(calls, ["validate", "rag", "openrouter"])
        self.assertIn(
            "Additional retrieved memory", generator.assert_request.system_instruction
        )
        self.assertIsNone(req.cache_name)
        vertex.stream_chat_completion.assert_not_called()

    async def test_omitted_provider_keeps_vertex_and_cache(self):
        from app.api.routes import chat_completions
        from app.services.graph_rag import GraphRagOutcome

        req = ChatCompletionRequest(
            model="gemini-3.1-pro-preview",
            messages=[],
            cache_name="cachedContents/unchanged",
        )

        class Vertex:
            async def stream_chat_completion(self, req, cache_manager=None):
                yield "gemini answer"

        alternative = AsyncMock()
        with patch(
            "app.api.routes.maybe_run_graph_rag",
            return_value=GraphRagOutcome(enabled=False, skip_reason="test"),
        ):
            response = await chat_completions(
                req, "secret", Vertex(), alternative, object(), object()
            )
            self.assertEqual(
                [chunk async for chunk in response.body_iterator], ["gemini answer"]
            )
        self.assertEqual(req.provider, "vertex")
        self.assertEqual(req.cache_name, "cachedContents/unchanged")
        alternative.validate_request.assert_not_called()
        alternative.stream_chat_completion.assert_not_called()
