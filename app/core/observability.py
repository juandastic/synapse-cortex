"""
Shared observability helpers for OpenTelemetry spans.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import httpx
from opentelemetry.trace import Span, Status, StatusCode

MAX_ERROR_MESSAGE_LEN = 300


def set_span_attributes(span: Span, attributes: Mapping[str, Any]) -> None:
    """Set non-empty primitive attributes on a span."""
    if not span.is_recording():
        return

    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(key, value)


def mark_span_success(span: Span, status: str = "success") -> None:
    """Mark span operation status as success/skipped."""
    if not span.is_recording():
        return
    span.set_attribute("operation.status", status)


def mark_span_error(
    span: Span,
    exc: Exception,
    *,
    category: str = "unknown",
    code: str = "UNEXPECTED_ERROR",
    status_description: str | None = None,
    extra_attributes: Mapping[str, Any] | None = None,
) -> None:
    """Record exception and normalized error attributes on a span."""
    if not span.is_recording():
        return

    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, status_description or str(exc)))
    span.set_attribute("operation.status", "failed")
    span.set_attribute("error.category", category)
    span.set_attribute("error.code", code)
    span.set_attribute("error.type", type(exc).__name__)
    span.set_attribute("error.message", truncate_error_message(str(exc)))

    if extra_attributes:
        set_span_attributes(span, extra_attributes)


def truncate_error_message(message: str | None, max_len: int = MAX_ERROR_MESSAGE_LEN) -> str:
    if not message:
        return ""
    if len(message) <= max_len:
        return message
    return f"{message[:max_len]}..."


def anonymize_id(value: str | None) -> str:
    """Return deterministic short hash for potentially sensitive IDs."""
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def classify_error(exc: Exception) -> tuple[str, str]:
    """Classify errors into category + code for consistent querying."""
    message = str(exc).lower()

    if isinstance(exc, httpx.TimeoutException) or "timeout" in message:
        return "timeout", "UPSTREAM_TIMEOUT"
    if isinstance(exc, httpx.NetworkError):
        return "network", "UPSTREAM_NETWORK_ERROR"
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 429:
            return "rate_limited", "UPSTREAM_RATE_LIMITED"
        if status_code is not None and 400 <= status_code < 500:
            return "upstream_4xx", "UPSTREAM_HTTP_4XX"
        if status_code is not None and status_code >= 500:
            return "upstream_5xx", "UPSTREAM_HTTP_5XX"
    if "429" in message or "rate limit" in message:
        return "rate_limited", "UPSTREAM_RATE_LIMITED"
    if "validation" in message or "invalid" in message:
        return "validation", "VALIDATION_ERROR"
    if "neo4j" in message or "cypher" in message or "database" in message:
        return "database", "DATABASE_ERROR"
    return "unknown", "UNEXPECTED_ERROR"


def extract_upstream_status_code(exc: Exception) -> int | None:
    """Best-effort extraction of upstream HTTP status code from SDK exceptions."""
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "status_code"):
        return response.status_code

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None
