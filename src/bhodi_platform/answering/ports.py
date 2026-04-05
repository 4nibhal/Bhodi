from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class RetrieverPort(Protocol):
    def invoke(self, query: str) -> list[Any]:
        """Return documents relevant to a query."""


class LanguageModelPort(Protocol):
    def invoke(self, messages: Sequence[Any]) -> Any:
        """Generate a response from the provided message sequence."""


class TokenizerPort(Protocol):
    def __call__(self, texts: Any, **kwargs: Any) -> Any:
        """Tokenize the provided texts."""

    def encode(self, text: str) -> Sequence[int]:
        """Encode a single text into token ids."""


class SummarizerPort(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Summarize a long text input."""


class RerankerPort(Protocol):
    def __call__(self, pair: tuple[str, str], **kwargs: Any) -> Any:
        """Score a query/document pair."""


class VectorStorePort(Protocol):
    def add_texts(self, texts: Sequence[str]) -> Any:
        """Persist texts into a vector store."""


class LogWriter(Protocol):
    def __call__(self, log_text: str) -> None:
        """Persist a log line."""
