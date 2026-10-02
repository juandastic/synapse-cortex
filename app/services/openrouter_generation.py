"""OpenRouter chat generation after the shared Cortex memory retrieval step."""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

import httpx
from fastapi import HTTPException
from opentelemetry import trace

from app.core.posthog import capture_generation
from app.schemas.models import ChatCompletionRequest

logger = logging.getLogger(__name__)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Explicit choices mirrored by the web/Convex chat catalog. No automatic model routing.
MODEL_CONFIG = {
    "openai/gpt-6.1-sol": {"reasoning": {"effort": "high"}, "images": True},
    "anthropic/claude-sonnet-5.5": {
        "reasoning": {"enabled": True, "effort": "high"},
        "images": True,
    },
    "qwen/qwen3.8-max-0902": {"reasoning": {"effort": "high"}, "images": True},
    "moonshotai/kimi-k2.6": {"reasoning": {"enabled": True}, "images": True},
}


class OpenRouterGenerationService:
    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(300, connect=15),
        )

    async def close(self):
        await self._client.aclose()

    def validate_request(self, request: ChatCompletionRequest):
        config = MODEL_CONFIG.get(request.model)
        if config is None:
            raise HTTPException(400, "Unsupported OpenRouter chat model")
        if not self._api_key:
            raise HTTPException(503, "OpenRouter chat generation is not configured")
        has_images = any(
            part.type == "image_url"
            for message in request.messages
            if isinstance(message.content, list)
            for part in message.content
        )
        if has_images and not config["images"]:
            raise HTTPException(
                400,
                "This model does not support images in the conversation. Choose a model with vision.",
            )

    def build_payload(self, request: ChatCompletionRequest) -> dict:
        self.validate_request(request)
        # Memory compilation is always sent as text; Gemini cache IDs never leave Cortex.
        system_parts = [request.system_instruction or "", request.compilation or ""]
        system_parts.extend(
            message.content
            for message in request.messages
            if message.role == "system" and isinstance(message.content, str)
        )
        messages = []
        system_text = "\n\n".join(part for part in system_parts if part)
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.extend(
            message.model_dump(exclude_none=True)
            for message in request.messages
            if message.role != "system"
        )
        return {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "reasoning": {**MODEL_CONFIG[request.model]["reasoning"], "exclude": True},
            # Provider failover can retain this model; it must support its reasoning options.
            "provider": {"require_parameters": True},
            "max_tokens": 16384,
        }

    async def stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        started_at = time.monotonic()
        model_used = request.model
        usage = {}
        input_messages = []
        answer_parts = []
        finish_reason = None
        response_chars = 0
        received_done = False
        with trace.get_tracer(__name__).start_as_current_span(
            "chat.openrouter.stream",
            attributes={
                "chat.provider": "openrouter",
                "chat.model": request.model,
                "chat.session_id": request.session_id or "",
            },
        ) as span:
            try:
                payload = self.build_payload(request)
                input_messages = payload["messages"]
                async with self._client.stream(
                    "POST",
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                ) as response:
                    if response.is_error:
                        # Do not log request headers or prompts, including the API key.
                        body = await response.aread()
                        try:
                            detail = (
                                json.loads(body).get("error", {}).get("message", "Request rejected")
                            )
                        except (ValueError, AttributeError):
                            detail = "Request rejected"
                        raise RuntimeError(
                            f"OpenRouter HTTP {response.status_code}: {str(detail)[:300]}"
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            received_done = True
                            break
                        chunk = json.loads(data)
                        if chunk.get("error"):
                            raise RuntimeError(
                                f"OpenRouter: {str(chunk['error'].get('message', 'Generation failed'))[:300]}"
                            )
                        model_used = chunk.get("model") or model_used
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            response_chars += len(content)
                            answer_parts.append(content)
                            yield self._chunk(
                                completion_id, created, model_used, {"content": content}
                            )
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                if not received_done or not finish_reason:
                    raise RuntimeError("OpenRouter stream ended before completion")
                if not response_chars:
                    raise RuntimeError(
                        "OpenRouter returned no visible answer. Retry the selected model."
                    )
                normalized_usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "thoughts_tokens": (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    ),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens"
                    ),
                    "cost": usage.get("cost"),
                    "cache_enabled": False,
                    "grounding_enabled": False,
                    "grounding_used": False,
                    **request.rag_usage_fields,
                }
                span.set_attribute("chat.response_chars", response_chars)
                span.set_attribute("chat.model_used", model_used)
                self._capture_generation(
                    request,
                    model_used,
                    input_messages,
                    answer_parts,
                    usage,
                    started_at,
                    finish_reason,
                )
                yield self._chunk(
                    completion_id,
                    created,
                    model_used,
                    {},
                    finish_reason,
                    normalized_usage,
                )
                yield "data: [DONE]\n\n"
            except Exception as error:  # noqa: BLE001 - Send failures through the already-open SSE response.
                span.record_exception(error)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                logger.warning(
                    "OpenRouter generation failed for model=%s: %s",
                    request.model,
                    error,
                )
                self._capture_generation(
                    request,
                    model_used,
                    input_messages,
                    answer_parts,
                    usage,
                    started_at,
                    "error",
                    error=str(error),
                )
                yield f"data: {json.dumps({'error': {'message': str(error), 'code': 502}})}\n\n"

    @staticmethod
    def _capture_generation(
        request,
        model,
        input_messages,
        answer_parts,
        usage,
        started_at,
        finish_reason,
        error=None,
    ):
        properties = {
            "$ai_latency": time.monotonic() - started_at,
            "$ai_total_cost_usd": usage.get("cost"),
            "$ai_total_tokens": usage.get("total_tokens"),
            "$ai_reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            "$ai_cache_read_input_tokens": (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
            "requested_model": request.model,
            "finish_reason": finish_reason,
            **request.rag_usage_fields,
        }
        try:
            capture_generation(
                request.user_id or "anonymous",
                request.posthog_trace_id,
                session_id=request.session_id,
                model=model,
                provider="openrouter",
                input_messages=input_messages,
                output="".join(answer_parts),
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                error=error,
                properties={key: value for key, value in properties.items() if value is not None},
            )
        except Exception:
            # Analytics must never interrupt chat generation.
            logger.warning("Could not capture OpenRouter generation analytics", exc_info=True)

    @staticmethod
    def _chunk(completion_id, created, model, delta, finish_reason=None, usage=None):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            chunk["usage"] = usage
        return f"data: {json.dumps(chunk)}\n\n"
