from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


class DocumentLoader(Protocol):
    def load(self) -> list[Any]:
        """Load raw documents from a source."""


class DocumentSplitter(Protocol):
    def split_documents(self, documents: Sequence[Any]) -> list[Any]:
        """Split documents into indexable fragments."""


class VectorStorePort(Protocol):
    def add_documents(self, documents: Sequence[Any]) -> Any:
        """Persist document fragments in a vector store."""

    def as_retriever(self, *, search_kwargs: dict[str, Any]) -> Any:
        """Expose a retriever view for the stored documents."""
