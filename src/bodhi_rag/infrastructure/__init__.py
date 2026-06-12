from __future__ import annotations

from typing import Any

__all__ = ["RuntimeRegistry"]


def __getattr__(name: str) -> Any:
    if name == "RuntimeRegistry":
        from bodhi_rag.infrastructure.runtime_registry import RuntimeRegistry

        return RuntimeRegistry
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
