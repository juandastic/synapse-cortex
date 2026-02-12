"""
OpenTelemetry telemetry setup for Synapse Cortex.

Exports traces to Axiom via OTLP. Auto-instruments FastAPI (inbound requests)
and httpx (outbound HTTP calls, including Graphiti's internal Gemini calls).

If AXIOM_API_TOKEN is not set, telemetry is disabled (no-op).
"""

import logging
import os

from fastapi import FastAPI

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Module-level reference for shutdown
_tracer_provider = None


def setup_telemetry(app: FastAPI) -> None:
    """
    Configure OpenTelemetry and instrument the FastAPI app.

    Instruments FastAPI (inbound requests) and httpx (outbound HTTP).
    No-ops if AXIOM_API_TOKEN is not set.
    """
    settings = get_settings()
    if not settings.axiom_api_token:
        logger.info("AXIOM_API_TOKEN not set, telemetry disabled")
        return

    # Exclude /health from tracing to avoid noise
    os.environ.setdefault("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", "/health")

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        global _tracer_provider

        resource = Resource(attributes={SERVICE_NAME: settings.otel_service_name})
        provider = TracerProvider(resource=resource)

        endpoint = f"https://{settings.axiom_domain.rstrip('/')}/v1/traces"
        otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={
                "Authorization": f"Bearer {settings.axiom_api_token}",
                "X-Axiom-Dataset": settings.axiom_dataset,
            },
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(provider)
        _tracer_provider = provider

        # Auto-instrument httpx first (outbound calls)
        HTTPXClientInstrumentor().instrument()

        # Auto-instrument FastAPI (inbound requests)
        FastAPIInstrumentor.instrument_app(app)

        logger.info(
            f"Telemetry enabled: service={settings.otel_service_name}, "
            f"dataset={settings.axiom_dataset}"
        )
    except ImportError as e:
        logger.warning(f"OpenTelemetry packages not installed: {e}")
    except Exception as e:
        logger.warning(f"Failed to setup telemetry: {e}")


def shutdown_telemetry() -> None:
    """Flush and shut down the tracer provider. Safe to call if telemetry was disabled."""
    global _tracer_provider
    if _tracer_provider is None:
        return
    try:
        from opentelemetry.sdk.trace import TracerProvider

        if isinstance(_tracer_provider, TracerProvider):
            _tracer_provider.force_flush()
            _tracer_provider.shutdown()
        _tracer_provider = None
        logger.info("Telemetry shutdown complete")
    except Exception as e:
        logger.warning(f"Error during telemetry shutdown: {e}")
    finally:
        _tracer_provider = None
