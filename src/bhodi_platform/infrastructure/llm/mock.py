"""
Mock LLM adapter for testing.

Returns hardcoded responses without network calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bhodi_platform.infrastructure.tracing import traced

if TYPE_CHECKING:
    from bhodi_platform.application.config import LLMConfig
    from bhodi_platform.domain.entities import ConversationTurn
    from bhodi_platform.ports.vector_store import RetrievedDocument


class MockLLMAdapter:
    """
    Fake LLM adapter that returns hardcoded responses.

    Useful for E2E tests that don't want external API calls.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @traced("mock.llm.generate")
    async def generate(
        self,
        prompt: str,
        **kwargs: str | int | float,
    ) -> str:
        """Return a hardcoded response."""
        return "This is a mock response."

    @traced("mock.llm.generate_with_context")
    async def generate_with_context(
        self,
        query: str,
        contexts: list[RetrievedDocument],
        **kwargs: str | int | float,
    ) -> str:
        """Return a mock response citing the contexts."""
        context_count = len(contexts)
        return (
            f"Mock answer based on {context_count} retrieved documents. "
            f"Query was: '{query}'"
        )
