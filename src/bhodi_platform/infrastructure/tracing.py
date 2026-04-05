"""Tracing decorators and utilities for Bhodi ports."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    pass

F = TypeVar("F", bound=Callable)


def traced(operation_name: str | None = None):
    """Decorator to trace a function call.

    Usage:
        @traced("embedding.documents")
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from bhodi_platform.infrastructure.telemetry import span

            name = operation_name or f"{func.__module__}.{func.__name__}"
            async with span(name):
                return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
