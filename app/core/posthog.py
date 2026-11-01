"""
PostHog LLM Analytics - Initialization and helper functions.

Manual trace, span, and generation events complement the wrapped GenAI client.
"""

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from posthog import Posthog

from app.core.config import Settings

logger = logging.getLogger(__name__)

_posthog_client: Posthog | None = None


def init_posthog(settings: Settings) -> Posthog | None:
    """Initialize the PostHog client. Returns None if API key is not configured."""
    global _posthog_client  # noqa: PLW0603
    if not settings.posthog_api_key:
        logger.info("PostHog LLM Analytics disabled (no POSTHOG_API_KEY)")
        return None

    _posthog_client = Posthog(
        project_api_key=settings.posthog_api_key,
        host=settings.posthog_host,
    )
    logger.info("PostHog LLM Analytics initialized")
    return _posthog_client


def shutdown_posthog() -> None:
    """Flush and shutdown the PostHog client."""
    if _posthog_client is not None:
        _posthog_client.flush()
        _posthog_client.shutdown()
        logger.info("PostHog client shut down")


def get_posthog() -> Posthog | None:
    """Get the global PostHog client instance."""
    return _posthog_client


def capture_trace(
    distinct_id: str,
    trace_id: str,
    *,
    name: str | None = None,
    session_id: str | None = None,
    properties: dict[str, Any] | None = None,
) -> None:
    """Capture an $ai_trace event to group related generations and spans."""
    if _posthog_client is None:
        return
    props: dict[str, Any] = {"$ai_trace_id": trace_id}
    if name:
        props["$ai_trace_name"] = name
    if session_id:
        props["$ai_session_id"] = session_id
    if properties:
        props.update(properties)
    _posthog_client.capture(distinct_id=distinct_id, event="$ai_trace", properties=props)


def capture_generation(
    distinct_id: str,
    trace_id: str,
    *,
    input_messages: list[dict[str, str]] | str | None = None,
    output: str | None = None,
    model: str | None = None,
    provider: str = "google",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: float | None = None,
    error: str | None = None,
    session_id: str | None = None,
    properties: dict[str, Any] | None = None,
) -> None:
    """Capture an $ai_generation event for a single LLM call."""
    if _posthog_client is None:
        return
    props: dict[str, Any] = {
        "$ai_trace_id": trace_id,
        "$ai_provider": provider,
    }
    if session_id:
        props["$ai_session_id"] = session_id
    if model:
        props["$ai_model"] = model
    if input_messages is not None:
        props["$ai_input"] = input_messages
    if output is not None:
        props["$ai_output_choices"] = [{"role": "assistant", "content": output}]
    if input_tokens is not None:
        props["$ai_input_tokens"] = input_tokens
    if output_tokens is not None:
        props["$ai_output_tokens"] = output_tokens
    if latency_ms is not None:
        props["$ai_latency"] = round(latency_ms, 2)
    if error:
        props["$ai_is_error"] = True
        props["$ai_error"] = error
    if properties:
        props.update(properties)
    _posthog_client.capture(distinct_id=distinct_id, event="$ai_generation", properties=props)


def capture_span(
    distinct_id: str,
    trace_id: str,
    *,
    name: str,
    input_data: str | list | None = None,
    output_data: str | list | None = None,
    duration_ms: float | None = None,
    properties: dict[str, Any] | None = None,
) -> None:
    """Capture an $ai_span event for a unit of work within a trace."""
    if _posthog_client is None:
        return
    props: dict[str, Any] = {
        "$ai_trace_id": trace_id,
        "$ai_span_name": name,
    }
    if input_data is not None:
        props["$ai_input"] = (
            input_data
            if isinstance(input_data, list)
            else [{"role": "user", "content": input_data}]
        )
    if output_data is not None:
        props["$ai_output_choices"] = (
            output_data
            if isinstance(output_data, list)
            else [{"role": "assistant", "content": output_data}]
        )
    if duration_ms is not None:
        props["$ai_latency"] = round(duration_ms, 2)
    if properties:
        props.update(properties)
    _posthog_client.capture(distinct_id=distinct_id, event="$ai_span", properties=props)


def new_trace_id() -> str:
    """Generate a new trace ID."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# GenAI client user context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMContext:
    distinct_id: str
    trace_id: str
    session_id: str | None = None


_llm_context: ContextVar[LLMContext | None] = ContextVar("llm_context", default=None)


def get_llm_context() -> LLMContext | None:
    """Identity inherited by child tasks, isolated from concurrent requests."""
    return _llm_context.get()


class ContextualGeminiModels:
    """Pass task-local identity into PostHog without changing SDK client defaults."""

    def __init__(self, models):
        self._models = models

    def __getattr__(self, name):
        return getattr(self._models, name)

    async def generate_content(self, **kwargs):
        context = get_llm_context()
        if context is not None:
            kwargs.setdefault("posthog_distinct_id", context.distinct_id)
            kwargs.setdefault("posthog_trace_id", context.trace_id)
            properties = {}
            if context.session_id:
                properties["$ai_session_id"] = context.session_id
            kwargs["posthog_properties"] = {**properties, **kwargs.get("posthog_properties", {})}
        return await self._models.generate_content(**kwargs)


@contextmanager
def posthog_user_context(
    distinct_id: str, trace_id: str | None = None, session_id: str | None = None
):
    """Associate internal LLM calls with a user, trace and session across async tasks."""
    token = _llm_context.set(LLMContext(distinct_id, trace_id or new_trace_id(), session_id))
    try:
        yield
    finally:
        _llm_context.reset(token)
