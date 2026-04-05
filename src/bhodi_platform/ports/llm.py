"""
LLM port definition.

Defines the contract for language model adapters.
"""

from __future__ import annotations

from typing import Protocol

from bhodi_platform.ports.vector_store import RetrievedDocument


class LLMPort(Protocol):
    """
    Protocol for language model generation.

    Adapters implementing this port handle generating
    text responses from LLMs.
    """

    async def generate(
        self,
        prompt: str,
        **kwargs: str | int | float,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate from.
            **kwargs: Provider-specific generation parameters.

        Returns:
            Generated text response.

        Raises:
            LLMError: If generation fails.
        """
        ...

    async def generate_with_context(
        self,
        query: str,
        contexts: list[RetrievedDocument],
        **kwargs: str | int | float,
    ) -> str:
        """
        Generate an answer given a query and retrieved context documents.

        Args:
            query: The user query.
            contexts: Retrieved documents to use as context.
            **kwargs: Provider-specific generation parameters.

        Returns:
            Generated answer text.

        Raises:
            LLMError: If generation fails.
        """
        ...
