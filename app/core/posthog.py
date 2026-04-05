"""
PostHog LLM Analytics - Initialization and helper functions.

Provides manual capture of $ai_trace, $ai_span, and $ai_generation events
for LLM observability in PostHog. Uses manual capture instead of the SDK
wrapper because the project relies on async streaming (client.aio) which
the PostHog GenAI wrapper does not yet support.
"""

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
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
    properties: dict[str, Any] | None = None,
) -> None:
    """Capture an $ai_generation event for a single LLM call."""
    if _posthog_client is None:
        return
    props: dict[str, Any] = {
        "$ai_trace_id": trace_id,
        "$ai_provider": provider,
    }
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
        props["$ai_input"] = input_data if isinstance(input_data, list) else [{"role": "user", "content": input_data}]
    if output_data is not None:
        props["$ai_output_choices"] = output_data if isinstance(output_data, list) else [{"role": "assistant", "content": output_data}]
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

_posthog_genai_client = None


def set_posthog_genai_client(client) -> None:
    """Store a reference to the PostHog-wrapped GenAI client for user context switching."""
    global _posthog_genai_client  # noqa: PLW0603
    _posthog_genai_client = client


@contextmanager
def posthog_user_context(distinct_id: str, trace_id: str | None = None):
    """Temporarily set the default distinct_id (and optionally trace_id) on the
    PostHog GenAI wrapper so that all LLM calls made within the block
    (including Graphiti's internal calls) are associated with the correct user.

    Usage:
        with posthog_user_context(user_id, trace_id):
            await graphiti.add_episode(...)
    """
    client = _posthog_genai_client
    if client is None or not hasattr(client, "models"):
        yield
        return

    models = client.models
    prev_distinct_id = getattr(models, "_default_distinct_id", None)
    prev_properties = getattr(models, "_default_properties", {})

    models._default_distinct_id = distinct_id
    if trace_id:
        models._default_properties = {**prev_properties, "$ai_trace_id": trace_id}
    try:
        yield
    finally:
        models._default_distinct_id = prev_distinct_id
        models._default_properties = prev_properties


@dataclass
class SpanTimer:
    """Simple timer for measuring span duration."""

    name: str
    _start: float = field(default_factory=time.monotonic, init=False)

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000
