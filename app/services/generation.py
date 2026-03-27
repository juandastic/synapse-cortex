"""
Generation Service - Gemini streaming with OpenAI-compatible SSE format.

Wraps the Google GenAI SDK and outputs SSE-formatted chunks
compatible with OpenAI's streaming API format.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

import httpx
from google import genai
from google.genai import types
from opentelemetry import trace

from app.core.observability import (
    classify_error,
    extract_upstream_status_code,
    mark_span_error,
    mark_span_success,
    set_span_attributes,
    truncate_error_message,
)
from app.schemas.models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatMessage,
    UsageData,
)

logger = logging.getLogger(__name__)


class GenerationService:
    """Service for generating chat completions using Google Gemini."""

    def __init__(self, client: genai.Client):
        self._client = client

    async def stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion in OpenAI-compatible SSE format.
        
        Args:
            request: The chat completion request.
        
        Yields:
            SSE-formatted strings: "data: {json}\n\n"
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        start_total = time.monotonic()
        first_chunk_latency_ms: float | None = None
        chunk_count = 0
        raw_chunk_count = 0
        response_chars = 0
        finish_reason = "stop"
        seen_finish_reasons: list[str] = []
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span(
            "chat.completion.stream",
            attributes={
                "chat.completion_id": completion_id,
                "chat.model": request.model,
                "chat.messages_count": len(request.messages),
            },
        ) as span:
            try:
                span.add_event("prepare_input")
                # Convert messages to Gemini format
                gemini_contents = await self._convert_to_gemini_format(request.messages)
                set_span_attributes(
                    span,
                    {
                        "chat.gemini_contents_count": len(gemini_contents),
                        "chat.phase": "request_upstream",
                    },
                )

                # Stream the first chunk with role
                first_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            delta=ChatCompletionDelta(role="assistant"),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {first_chunk.model_dump_json()}\n\n"

                # Track usage_metadata across chunks (fully populated on last chunk)
                usage_metadata = None

                # Use async client for true async streaming
                span.add_event("request_upstream")
                async for chunk in await self._client.aio.models.generate_content_stream(
                    model=request.model,
                    contents=gemini_contents,
                ):
                    set_span_attributes(span, {"chat.phase": "stream_chunks"})
                    if first_chunk_latency_ms is None:
                        first_chunk_latency_ms = (time.monotonic() - start_total) * 1000
                        set_span_attributes(
                            span,
                            {
                                "chat.upstream_first_chunk_ms": round(first_chunk_latency_ms, 2),
                            },
                        )

                    raw_chunk_count += 1

                    # Capture usage_metadata from each chunk; the last one is definitive
                    if chunk.usage_metadata:
                        usage_metadata = chunk.usage_metadata

                    chunk_finish_reason = self._extract_finish_reason(chunk)
                    if chunk_finish_reason:
                        finish_reason = chunk_finish_reason
                        if chunk_finish_reason not in seen_finish_reasons:
                            seen_finish_reasons.append(chunk_finish_reason)
                            # Emit an event for any non-stop finish reason so it's
                            # immediately visible in the trace timeline.
                            if chunk_finish_reason not in ("stop", "unspecified"):
                                span.add_event(
                                    "non_stop_finish_reason",
                                    {"finish_reason": chunk_finish_reason},
                                )

                    if chunk.text:
                        chunk_count += 1
                        response_chars += len(chunk.text)
                        content_chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=request.model,
                            choices=[
                                ChatCompletionChoice(
                                    index=0,
                                    delta=ChatCompletionDelta(content=chunk.text),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {content_chunk.model_dump_json()}\n\n"

                # Build usage data from the final chunk's metadata
                usage_data = None
                if usage_metadata:
                    usage_data = UsageData(
                        prompt_tokens=usage_metadata.prompt_token_count or 0,
                        completion_tokens=usage_metadata.candidates_token_count or 0,
                        total_tokens=usage_metadata.total_token_count or 0,
                        thoughts_tokens=getattr(usage_metadata, "thoughts_token_count", None),
                        cached_tokens=getattr(usage_metadata, "cached_content_token_count", None),
                        **request.rag_usage_fields,
                    )
                    set_span_attributes(
                        span,
                        {
                            "chat.tokens.prompt": usage_data.prompt_tokens,
                            "chat.tokens.completion": usage_data.completion_tokens,
                            "chat.tokens.total": usage_data.total_tokens,
                            "chat.tokens.thoughts": usage_data.thoughts_tokens,
                            "chat.tokens.cached": usage_data.cached_tokens,
                        },
                    )

                # Send final chunk with finish_reason and usage
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            delta=ChatCompletionDelta(),
                            finish_reason=finish_reason,
                        )
                    ],
                    usage=usage_data,
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"

                # Send the done marker
                yield "data: [DONE]\n\n"
                empty_response = chunk_count == 0 and raw_chunk_count > 0
                thoughts_tokens = (
                    getattr(usage_metadata, "thoughts_token_count", None)
                    if usage_metadata
                    else None
                )
                # Distinguish None (model doesn't support thinking) from 0
                # (thinking model stopped before generating any thoughts — hard stop signal).
                thoughts_tokens_reported = thoughts_tokens is not None
                empty_response_type = None
                if empty_response:
                    if thoughts_tokens_reported and thoughts_tokens == 0:
                        empty_response_type = "pre_generation_hard_stop"
                    elif thoughts_tokens_reported and thoughts_tokens > 0:
                        empty_response_type = "thinking_only"
                    else:
                        empty_response_type = "unknown"

                set_span_attributes(
                    span,
                    {
                        "chat.phase": "finalize",
                        "chat.finish_reason": finish_reason,
                        "chat.finish_reasons_seen": ",".join(seen_finish_reasons) if seen_finish_reasons else finish_reason,
                        "chat.stream_chunks_count": chunk_count,
                        "chat.stream_raw_chunks_count": raw_chunk_count,
                        "chat.response_chars": response_chars,
                        "chat.empty_response": empty_response,
                        "chat.empty_response_type": empty_response_type,
                        "chat.total_duration_ms": round((time.monotonic() - start_total) * 1000, 2),
                    },
                )
                # Emit a dedicated event when the model responded but sent no text —
                # makes empty-response traces easy to filter and alert on.
                if empty_response:
                    span.add_event(
                        "empty_response_detected",
                        {
                            "raw_chunks": raw_chunk_count,
                            "finish_reason": finish_reason,
                            "prompt_tokens": usage_metadata.prompt_token_count if usage_metadata else 0,
                            "thoughts_tokens": thoughts_tokens if thoughts_tokens is not None else -1,
                            "thoughts_tokens_reported": thoughts_tokens_reported,
                            "empty_response_type": empty_response_type,
                        },
                    )
                mark_span_success(span)

            except Exception as e:
                category, code = classify_error(e)
                upstream_status_code = extract_upstream_status_code(e)
                mark_span_error(
                    span,
                    e,
                    category=category,
                    code="CHAT_STREAM_ERROR" if code == "UNEXPECTED_ERROR" else code,
                    extra_attributes={
                        "chat.phase": "error",
                        "chat.stream_chunks_count": chunk_count,
                        "chat.response_chars": response_chars,
                        "chat.total_duration_ms": round((time.monotonic() - start_total) * 1000, 2),
                        "upstream.status_code": upstream_status_code,
                        "upstream.error_type": type(e).__name__,
                        "upstream.error_message": truncate_error_message(str(e)),
                    },
                )
                logger.error(f"Error during generation: {e}", exc_info=True)
                # Send error as a chunk
                error_payload = {
                    "error": {
                        "message": truncate_error_message(str(e)),
                        "code": "CHAT_STREAM_ERROR" if code == "UNEXPECTED_ERROR" else code,
                        "category": category,
                        "status_code": upstream_status_code,
                    }
                }
                yield f"data: {json.dumps(error_payload)}\n\n"
                yield "data: [DONE]\n\n"

    async def _download_image(self, url: str) -> tuple[bytes, str]:
        """
        Download image from a URL and return (bytes, mime_type).

        Uses the Content-Type response header to detect MIME type,
        falling back to image/jpeg if unavailable.
        """
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("chat.image.download") as span:
            start = time.monotonic()
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    mime_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
                    set_span_attributes(
                        span,
                        {
                            "chat.image.mime_type": mime_type,
                            "chat.image.size_bytes": len(response.content),
                            "chat.image.download_ms": round((time.monotonic() - start) * 1000, 2),
                        },
                    )
                    mark_span_success(span)
                    return response.content, mime_type
            except Exception as e:
                category, code = classify_error(e)
                mark_span_error(
                    span,
                    e,
                    category=category,
                    code="CHAT_IMAGE_DOWNLOAD_ERROR" if code == "UNEXPECTED_ERROR" else code,
                    extra_attributes={
                        "chat.image.download_ms": round((time.monotonic() - start) * 1000, 2),
                    },
                )
                raise

    @staticmethod
    def _extract_finish_reason(chunk: object) -> str | None:
        """Best-effort extraction of finish_reason from Gemini chunk candidates."""
        candidates = getattr(chunk, "candidates", None)
        if not candidates:
            return None

        first = candidates[0]
        reason = getattr(first, "finish_reason", None)
        if reason is None:
            return None

        # Gemini SDK sometimes returns enums with a .name attribute.
        if hasattr(reason, "name"):
            return str(reason.name).lower()
        return str(reason).lower()

    async def _build_parts(self, content: str | list) -> list[types.Part]:
        """
        Convert a message's content (string or multimodal array) into
        a list of Gemini Part objects.
        """
        if isinstance(content, str):
            return [types.Part.from_text(text=content)]

        parts: list[types.Part] = []
        for item in content:
            if item.type == "text":
                parts.append(types.Part.from_text(text=item.text))
            elif item.type == "image_url":
                image_bytes, mime_type = await self._download_image(
                    item.image_url.url
                )
                parts.append(
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                )
        return parts

    async def _convert_to_gemini_format(
        self, messages: list[ChatMessage]
    ) -> list[types.Content]:
        """
        Convert OpenAI-style messages to Gemini format.

        Gemini uses a different format:
        - System messages become part of the first user message
        - Messages are grouped into user/model turns
        - Multimodal content arrays are converted to multiple Part objects
        """
        gemini_contents: list[types.Content] = []
        system_prompt: str | None = None

        # Extract system prompt if present (always a string)
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content if isinstance(msg.content, str) else None
                break

        # Build conversation history
        for msg in messages:
            if msg.role == "system":
                continue

            parts = await self._build_parts(msg.content)

            # Prepend system prompt to first user message
            if msg.role == "user" and system_prompt and not gemini_contents:
                parts.insert(0, types.Part.from_text(text=system_prompt))
                system_prompt = None  # Only prepend once

            role = "user" if msg.role == "user" else "model"
            gemini_contents.append(
                types.Content(role=role, parts=parts)
            )

        return gemini_contents
