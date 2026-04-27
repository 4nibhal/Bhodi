"""
Dependency injection container for new architecture.

Wires ports to adapter implementations based on configuration.
This is the new Container that works with the Protocol-based ports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bhodi_platform.ports.chunker import ChunkerPort
from bhodi_platform.ports.conversation_memory import ConversationMemoryPort
from bhodi_platform.ports.document_parser import DocumentParserPort
from bhodi_platform.ports.embedding import EmbeddingPort
from bhodi_platform.ports.llm import LLMPort
from bhodi_platform.ports.vector_store import VectorStorePort

if TYPE_CHECKING:
    from bhodi_platform.application.config import BhodiConfig
    from bhodi_platform.application.facade import BhodiApplication


class Container:
    """
    Dependency injection container for new Bhodi architecture.

    Builds adapter instances based on configuration and wires
    them to the BhodiApplication facade.
    """

    def __init__(self, config: BhodiConfig) -> None:
        self._config = config
        self._adapters: dict[type, object] = {}

    def build(self) -> BhodiApplication:
        """Build and return a fully wired BhodiApplication."""
        from bhodi_platform.application.facade import BhodiApplication

        return BhodiApplication(
            embedding=self._get_adapter(EmbeddingPort),
            vector_store=self._get_adapter(VectorStorePort),
            chunker=self._get_adapter(ChunkerPort),
            document_parser=self._get_adapter(DocumentParserPort),
            llm=self._get_adapter(LLMPort),
            conversation_memory=self._get_adapter(ConversationMemoryPort),
        )

    def _get_adapter(self, port_type: type) -> object:
        """Get or create adapter for given port type."""
        if port_type not in self._adapters:
            factory_name = f"_create_{port_type.__name__.replace('Port', '').lower()}_adapter"
            factory = getattr(self, factory_name, None)
            if factory is None:
                raise ValueError(f"No factory for adapter type: {port_type.__name__}")
            self._adapters[port_type] = factory()
        return self._adapters[port_type]

    def _create_embedding_adapter(self):
        """Create embedding adapter based on config."""
        from bhodi_platform.infrastructure.embedding.mock import MockEmbeddingAdapter
        from bhodi_platform.ports.embedding import EmbeddingPort

        provider = self._config.embedding.provider.lower()

        if provider == "mock":
            return MockEmbeddingAdapter(self._config.embedding)

        if provider == "openai":
            from bhodi_platform.infrastructure.embedding.openai import (
                OpenAIEmbeddingsAdapter,
            )

            return OpenAIEmbeddingsAdapter(self._config.embedding)

        # Default to mock if provider not recognized
        return MockEmbeddingAdapter(self._config.embedding)

    def _create_vectorstore_adapter(self):
        """Create vector store adapter based on config."""
        from bhodi_platform.infrastructure.vector_store.in_memory import (
            MockVectorStoreAdapter,
        )
        from bhodi_platform.ports.vector_store import VectorStorePort

        provider = self._config.vector_store.provider.lower()

        if provider == "in_memory":
            return MockVectorStoreAdapter(self._config.vector_store)

        if provider == "chroma":
            from bhodi_platform.infrastructure.vector_store.chroma import (
                ChromaVectorStoreAdapter,
            )

            return ChromaVectorStoreAdapter(self._config.vector_store)

        # Default to in_memory if provider not recognized
        return MockVectorStoreAdapter(self._config.vector_store)

    def _create_chunker_adapter(self):
        """Create chunker adapter based on config."""
        from bhodi_platform.infrastructure.chunker.fixed_size import (
            FixedSizeChunkerAdapter,
        )
        from bhodi_platform.infrastructure.chunker.recursive import (
            RecursiveChunkerAdapter,
        )
        from bhodi_platform.ports.chunker import ChunkerPort

        provider = self._config.chunker.provider.lower()

        if provider == "fixed_size":
            return FixedSizeChunkerAdapter(self._config.chunker)

        if provider == "recursive":
            return RecursiveChunkerAdapter(self._config.chunker)

        # Default to fixed_size if provider not recognized
        return FixedSizeChunkerAdapter(self._config.chunker)

    def _create_documentparser_adapter(self):
        """Create document parser adapter based on config."""
        from bhodi_platform.infrastructure.document_parser.mock import (
            MockDocumentParserAdapter,
        )
        from bhodi_platform.ports.document_parser import DocumentParserPort

        provider = self._config.parser.provider.lower()

        if provider == "mock":
            return MockDocumentParserAdapter(self._config.parser)

        if provider == "pypdf":
            from bhodi_platform.infrastructure.document_parser.pypdf import (
                PyPDFDocumentParserAdapter,
            )

            return PyPDFDocumentParserAdapter(self._config.parser)

        # Default to mock if provider not recognized
        return MockDocumentParserAdapter(self._config.parser)

    def _create_llm_adapter(self):
        """Create LLM adapter based on config."""
        from bhodi_platform.infrastructure.llm.mock import MockLLMAdapter
        from bhodi_platform.ports.llm import LLMPort

        provider = self._config.llm.provider.lower()

        if provider == "mock":
            return MockLLMAdapter(self._config.llm)

        if provider == "ollama":
            from bhodi_platform.infrastructure.llm.ollama import OllamaLLMAdapter

            return OllamaLLMAdapter(self._config.llm)

        if provider == "openai":
            from bhodi_platform.infrastructure.llm.openai import OpenAILLMAdapter

            return OpenAILLMAdapter(self._config.llm)

        # Default to mock if provider not recognized
        return MockLLMAdapter(self._config.llm)

    def _create_conversationmemory_adapter(self):
        """Create conversation memory adapter based on config."""
        from bhodi_platform.infrastructure.conversation_memory.volatile import (
            VolatileConversationMemoryAdapter,
        )
        from bhodi_platform.ports.conversation_memory import ConversationMemoryPort

        provider = self._config.conversation.provider.lower()

        if provider == "volatile":
            return VolatileConversationMemoryAdapter(self._config.conversation)

        # Default to volatile if provider not recognized
        return VolatileConversationMemoryAdapter(self._config.conversation)
