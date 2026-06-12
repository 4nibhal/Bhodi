"""OpenTelemetry instrumentation for bodhi-rag."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from opentelemetry.trace import Span, Tracer


class _TelemetryState:
    """Encapsulates mutable module-level telemetry state to avoid `global`."""

    tracer: Tracer | None = None
    enabled: bool = True


_state = _TelemetryState()


def get_tracer() -> Tracer:
    """Get or create the tracer."""
    if _state.tracer is None:
        from opentelemetry import trace

        _state.tracer = trace.get_tracer("bodhi-rag")
    return _state.tracer


def set_enabled(*, enabled: bool) -> None:
    """Enable or disable telemetry."""
    _state.enabled = enabled


@asynccontextmanager
async def span(
    name: str,
    attributes: dict[str, str | int | float] | None = None,
) -> AsyncIterator[Span | None]:
    """
    Create an async span for telemetry.

    Usage:
        async with span("indexing.embed", {"document_id": doc_id}):
            embeddings = await adapter.embed_documents(texts)
    """
    if not _state.enabled:
        yield None
        return

    tracer = get_tracer()
    with tracer.start_as_current_span(name) as current_span:
        if attributes:
            for key, value in attributes.items():
                current_span.set_attribute(key, value)
        yield current_span
