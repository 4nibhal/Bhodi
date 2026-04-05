"""OpenTelemetry instrumentation for Bhodi."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.trace import Span
    from opentelemetry.trace import Tracer

_tracer: Tracer | None = None
_enabled: bool = True


def get_tracer() -> Tracer:
    """Get or create the tracer."""
    global _tracer
    if _tracer is None:
        from opentelemetry import trace

        _tracer = trace.get_tracer("bhodi")
    return _tracer


def set_enabled(enabled: bool) -> None:
    """Enable or disable telemetry."""
    global _enabled
    _enabled = enabled


@asynccontextmanager
async def span(name: str, attributes: dict[str, str | int | float] | None = None):
    """Create an async span for telemetry.

    Usage:
        async with span("indexing.embed", {"document_id": doc_id}):
            embeddings = await adapter.embed_documents(texts)
    """
    if not _enabled:
        yield None
        return

    tracer = get_tracer()
    with tracer.start_as_current_span(name) as current_span:
        if attributes:
            for key, value in attributes.items():
                current_span.set_attribute(key, value)
        yield current_span
