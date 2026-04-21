"""
Generation Service - Gemini streaming with OpenAI-compatible SSE format.

Wraps the Google GenAI SDK and outputs SSE-formatted chunks
compatible with OpenAI's streaming API format.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

import httpx
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
from app.services.cache_manager import CacheManager


logger = logging.getLogger(__name__)

# Strong refs for fire-and-forget background tasks. asyncio only keeps weak
# references, so without this the GC can drop tasks mid-execution.
_background_tasks: set[asyncio.Task] = set()


def _spawn_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


class GenerationService:
    """Service for generating chat completions using Google Gemini."""

    def __init__(self, client):
        self._client = client
        # Detect if the client is a PostHog-wrapped AsyncClient
        self._is_posthog_client = hasattr(client, "models") and hasattr(
            client.models, "_ph_client"
        )

    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        cache_manager: CacheManager | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion in OpenAI-compatible SSE format.

        Args:
            request: The chat completion request.
            cache_manager: CacheManager for invalidating stale caches on fallback.

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
                "cache.enabled": request.cache_name is not None,
                "cache.name": request.cache_name or "",
            },
        ) as span:
            try:
                span.add_event("prepare_input")
                # When cache_name is present, the compilation is already in
                # the Gemini cache as system_instruction — don't re-send it.
                use_cache = request.cache_name is not None
                active_cache_name = request.cache_name
                gemini_contents = await self._build_contents(
                    request, inline_compilation=not use_cache,
                )
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

                # Use async streaming — route through PostHog wrapper or raw client.
                # The Google SDK is lazy: the HTTP request only happens on first
                # iteration, so cache errors surface when we pull the first chunk,
                # not when we `await` the stream factory. We peek the first chunk
                # under a try/except to enable cache-error fallback BEFORE any
                # bytes are yielded to the client.
                span.add_event("request_upstream")

                async def _open_and_peek(cache_name_to_use: str | None):
                    s = await self._stream_gemini(
                        request.model, gemini_contents, request.user_id,
                        trace_id=request.posthog_trace_id,
                        session_id=request.session_id,
                        cache_name=cache_name_to_use,
                    )
                    iterator = s.__aiter__()
                    try:
                        first = await iterator.__anext__()
                    except StopAsyncIteration:
                        first = None
                    return iterator, first

                # Fire-and-forget TTL bump alongside the upstream call so
                # idle gaps between a user's requests don't silently expire
                # the cache. Runs concurrently — no added latency.
                if use_cache and cache_manager is not None and active_cache_name:
                    _spawn_background(
                        cache_manager.refresh_ttl(active_cache_name)
                    )

                try:
                    stream_iter, first_chunk = await _open_and_peek(active_cache_name)
                except Exception as open_err:
                    # Any failure on the first cached upstream call falls back
                    # to a no-cache retry. The fallback also recovers from
                    # non-cache errors (e.g. transient 5xx, model resolution
                    # blips) at the cost of one extra round-trip when the
                    # second call fails the same way. Trace attributes
                    # (cache.fallback_triggered / cache.fallback_error) make
                    # this visible in Axiom regardless of root cause.
                    if use_cache:
                        logger.warning(
                            "Gemini cached call failed, falling back to full prompt "
                            "(cache=%s): %s",
                            active_cache_name, open_err,
                        )
                        request.cache_fallback_triggered = True
                        request.cache_fallback_error = truncate_error_message(str(open_err))
                        if cache_manager is not None and active_cache_name:
                            await cache_manager.invalidate_by_name(active_cache_name)
                        span.add_event(
                            "cache_fallback_to_full_prompt",
                            {
                                "error": request.cache_fallback_error,
                                "original_cache_name": active_cache_name or "",
                            },
                        )
                        # Rebuild contents with the compilation inlined and retry.
                        use_cache = False
                        active_cache_name = None
                        gemini_contents = await self._build_contents(
                            request, inline_compilation=True,
                        )
                        stream_iter, first_chunk = await _open_and_peek(None)
                    else:
                        raise

                async def _chunks():
                    if first_chunk is not None:
                        yield first_chunk
                    async for c in stream_iter:
                        yield c

                async for chunk in _chunks():
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
                    cached_tokens = getattr(usage_metadata, "cached_content_token_count", None)
                    prompt_tokens = usage_metadata.prompt_token_count or 0
                    cache_hit = bool(cached_tokens and cached_tokens > 0)
                    usage_data = UsageData(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=usage_metadata.candidates_token_count or 0,
                        total_tokens=usage_metadata.total_token_count or 0,
                        thoughts_tokens=getattr(usage_metadata, "thoughts_token_count", None),
                        cached_tokens=cached_tokens,
                        cache_enabled=use_cache,
                        cache_hit=cache_hit,
                        cache_fallback_triggered=request.cache_fallback_triggered,
                        **request.rag_usage_fields,
                    )
                    hit_ratio = (
                        (cached_tokens or 0) / prompt_tokens if prompt_tokens > 0 else 0.0
                    )
                    # Refresh TTL on hit so active users don't hit expiration
                    # mid-session (Gemini caches expire by wallclock, not usage).
                    if (
                        cache_hit
                        and active_cache_name
                        and cache_manager is not None
                    ):
                        _spawn_background(
                            cache_manager.refresh_ttl(active_cache_name)
                        )
                    set_span_attributes(
                        span,
                        {
                            "chat.tokens.prompt": usage_data.prompt_tokens,
                            "chat.tokens.completion": usage_data.completion_tokens,
                            "chat.tokens.total": usage_data.total_tokens,
                            "chat.tokens.thoughts": usage_data.thoughts_tokens,
                            "chat.tokens.cached": usage_data.cached_tokens,
                            "cache.hit": cache_hit,
                            "cache.hit_ratio": round(hit_ratio, 4),
                            "cache.fallback_triggered": request.cache_fallback_triggered,
                            "cache.fallback_error": request.cache_fallback_error,
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

    async def _stream_gemini(
        self, model: str, contents: list, user_id: str | None,
        trace_id: str = "", session_id: str | None = None,
        cache_name: str | None = None,
    ):
        """Route streaming through PostHog wrapper (auto-tracked) or raw client.

        When ``cache_name`` is provided, ``cached_content`` is passed to Gemini
        so the cached system_instruction is used as a prefix.
        """
        config = (
            types.GenerateContentConfig(cached_content=cache_name)
            if cache_name else None
        )

        if self._is_posthog_client:
            posthog_props: dict[str, str] = {}
            if session_id:
                posthog_props["$ai_session_id"] = session_id
            kwargs: dict = {
                "model": model,
                "contents": contents,
                "posthog_distinct_id": user_id or "anonymous",
                "posthog_trace_id": trace_id,
            }
            if posthog_props:
                kwargs["posthog_properties"] = posthog_props
            if config is not None:
                kwargs["config"] = config
            return await self._client.models.generate_content_stream(**kwargs)

        raw_kwargs: dict = {"model": model, "contents": contents}
        if config is not None:
            raw_kwargs["config"] = config
        return await self._client.aio.models.generate_content_stream(**raw_kwargs)

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

    async def _build_contents(
        self,
        request: ChatCompletionRequest,
        *,
        inline_compilation: bool,
    ) -> list[types.Content]:
        """Convert a ChatCompletionRequest into Gemini ``Content`` objects.

        Accepts both the new shape (``system_instruction`` + ``compilation``
        fields) and the legacy shape (everything in a ``role=system`` message).
        When ``inline_compilation`` is False, the compilation is omitted (it
        lives in the Gemini cache as system_instruction).
        """
        system_prompt: str | None = request.system_instruction
        compilation: str | None = request.compilation

        # Legacy fallback: extract the first role=system message when no
        # explicit system_instruction was provided.
        if system_prompt is None:
            for msg in request.messages:
                if msg.role == "system" and isinstance(msg.content, str):
                    system_prompt = msg.content
                    break

        effective_system = system_prompt or ""
        if inline_compilation and compilation:
            effective_system = (
                f"{effective_system}\n\n{compilation}" if effective_system else compilation
            )
        effective_system = effective_system or None

        gemini_contents: list[types.Content] = []
        pending_system = effective_system

        for msg in request.messages:
            if msg.role == "system":
                continue

            parts = await self._build_parts(msg.content)

            # Prepend the effective system prompt to the first user message
            if msg.role == "user" and pending_system and not gemini_contents:
                parts.insert(0, types.Part.from_text(text=pending_system))
                pending_system = None  # Only prepend once

            role = "user" if msg.role == "user" else "model"
            gemini_contents.append(
                types.Content(role=role, parts=parts)
            )

        return gemini_contents
