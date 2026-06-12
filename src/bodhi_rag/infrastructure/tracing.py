"""Tracing decorators and utilities for bodhi-rag ports."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def traced(operation_name: str | None = None) -> Callable[[F], F]:
    """
    Trace a function call.

    Usage:
        @traced("embedding.documents")
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            from bodhi_rag.infrastructure.telemetry import span

            name = operation_name or f"{func.__module__}.{func.__name__}"
            async with span(name):
                return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
