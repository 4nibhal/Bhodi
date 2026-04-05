"""Contract tests verifying adapters fulfill their Protocol contracts."""

import pytest
from typing import get_type_hints

from bhodi_platform.application.config import (
    EmbeddingConfig,
    VectorStoreConfig,
    ChunkerConfig,
    LLMConfig,
    ConversationConfig,
)
from bhodi_platform.ports.embedding import EmbeddingPort
from bhodi_platform.ports.vector_store import VectorStorePort
from bhodi_platform.ports.chunker import ChunkerPort
from bhodi_platform.ports.llm import LLMPort
from bhodi_platform.ports.conversation_memory import ConversationMemoryPort
from bhodi_platform.infrastructure.embedding.mock import MockEmbeddingAdapter
from bhodi_platform.infrastructure.vector_store.in_memory import MockVectorStoreAdapter
from bhodi_platform.infrastructure.chunker.fixed_size import FixedSizeChunkerAdapter
from bhodi_platform.infrastructure.llm.mock import MockLLMAdapter
from bhodi_platform.infrastructure.conversation_memory.volatile import (
    VolatileConversationMemoryAdapter,
)


def _check_protocol_methods(adapter, protocol):
    """Verify adapter has all methods required by protocol."""
    required_methods = set()
    for name in dir(protocol):
        if not name.startswith("_"):
            obj = getattr(protocol, name, None)
            if callable(obj) or isinstance(obj, property):
                required_methods.add(name)

    missing = []
    for method in required_methods:
        if not hasattr(adapter, method):
            missing.append(method)
    return missing


class TestEmbeddingPortContract:
    """Verify MockEmbeddingAdapter fulfills EmbeddingPort."""

    @pytest.fixture
    def adapter(self):
        config = EmbeddingConfig(provider="mock", dimensions=384)
        return MockEmbeddingAdapter(config)

    def test_has_required_methods(self, adapter):
        """Adapter has all methods required by EmbeddingPort."""
        missing = _check_protocol_methods(adapter, EmbeddingPort)
        assert not missing, f"Missing methods: {missing}"

    @pytest.mark.asyncio
    async def test_embed_documents_returns_list_of_lists(self, adapter):
        """embed_documents returns list[list[float]]."""
        texts = ["hello", "world"]
        result = await adapter.embed_documents(texts)
        assert isinstance(result, list)
        assert len(result) == 2
        for embedding in result:
            assert isinstance(embedding, list)
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_query_returns_single_vector(self, adapter):
        """embed_query returns list[float]."""
        result = await adapter.embed_query("hello")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_dimensions(self, adapter):
        """dimensions returns int."""
        dims = await adapter.dimensions()
        assert isinstance(dims, int)
        assert dims == 384


class TestVectorStorePortContract:
    """Verify MockVectorStoreAdapter fulfills VectorStorePort."""

    @pytest.fixture
    def adapter(self):
        config = VectorStoreConfig(provider="in_memory")
        return MockVectorStoreAdapter(config)

    def test_has_required_methods(self, adapter):
        """Adapter has all methods required by VectorStorePort."""
        missing = _check_protocol_methods(adapter, VectorStorePort)
        assert not missing, f"Missing methods: {missing}"

    @pytest.mark.asyncio
    async def test_add_and_search(self, adapter):
        """Can add chunks and search."""
        from bhodi_platform.domain.entities import Chunk
        from bhodi_platform.domain.value_objects import ChunkId, DocumentId

        doc_id = DocumentId()
        chunk_id = ChunkId(document_id=doc_id, chunk_index=0)
        chunk = Chunk(
            id=chunk_id,
            document_id=doc_id,
            content="Test content",
            chunk_index=0,
            total_chunks=1,
        )

        await adapter.add([chunk], [[0.1, 0.2, 0.3]])
        results = await adapter.search([0.1, 0.2, 0.3], top_k=1)

        assert len(results) == 1
        assert results[0].text == "Test content"


class TestChunkerPortContract:
    """Verify FixedSizeChunkerAdapter fulfills ChunkerPort."""

    @pytest.fixture
    def adapter(self):
        config = ChunkerConfig(provider="fixed_size", chunk_size=100)
        return FixedSizeChunkerAdapter(config)

    def test_has_required_methods(self, adapter):
        """Adapter has all methods required by ChunkerPort."""
        missing = _check_protocol_methods(adapter, ChunkerPort)
        assert not missing, f"Missing methods: {missing}"

    def test_default_chunk_size(self, adapter):
        """default_chunk_size property."""
        assert isinstance(adapter.default_chunk_size, int)
        assert adapter.default_chunk_size == 100

    def test_default_overlap(self, adapter):
        """default_overlap property."""
        assert isinstance(adapter.default_overlap, int)


class TestLLMPortContract:
    """Verify MockLLMAdapter fulfills LLMPort."""

    @pytest.fixture
    def adapter(self):
        config = LLMConfig(provider="mock")
        return MockLLMAdapter(config)

    def test_has_required_methods(self, adapter):
        """Adapter has all methods required by LLMPort."""
        missing = _check_protocol_methods(adapter, LLMPort)
        assert not missing, f"Missing methods: {missing}"

    @pytest.mark.asyncio
    async def test_generate(self, adapter):
        """generate returns str."""
        result = await adapter.generate("Hello?")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_with_context(self, adapter):
        """generate_with_context returns str."""
        from bhodi_platform.domain.entities import RetrievedDocument
        from bhodi_platform.domain.value_objects import ChunkId, DocumentId

        doc_id = DocumentId()
        chunk_id = ChunkId(document_id=doc_id, chunk_index=0)
        retrieved = RetrievedDocument(
            chunk_id=chunk_id,
            document_id=doc_id,
            text="Context content",
            score=0.9,
        )

        result = await adapter.generate_with_context(
            "What is this?",
            [retrieved],
        )
        assert isinstance(result, str)


class TestConversationMemoryPortContract:
    """Verify VolatileConversationMemoryAdapter fulfills ConversationMemoryPort."""

    @pytest.fixture
    def adapter(self):
        config = ConversationConfig(provider="volatile")
        return VolatileConversationMemoryAdapter(config)

    def test_has_required_methods(self, adapter):
        """Adapter has all methods required by ConversationMemoryPort."""
        missing = _check_protocol_methods(adapter, ConversationMemoryPort)
        assert not missing, f"Missing methods: {missing}"

    @pytest.mark.asyncio
    async def test_add_and_get_history(self, adapter):
        """Can add turn and retrieve history."""
        from bhodi_platform.domain.entities import ConversationTurn
        from bhodi_platform.domain.value_objects import ConversationId

        conv_id = ConversationId()
        turn = ConversationTurn(
            conversation_id=conv_id,
            user_message="Hello",
            assistant_message="Hi there",
        )

        await adapter.add(conv_id, turn)
        history = await adapter.get_history(conv_id)

        assert len(history) == 1
        assert history[0].user_message == "Hello"

    @pytest.mark.asyncio
    async def test_clear(self, adapter):
        """Can clear conversation history."""
        from bhodi_platform.domain.entities import ConversationTurn
        from bhodi_platform.domain.value_objects import ConversationId

        conv_id = ConversationId()
        turn = ConversationTurn(
            conversation_id=conv_id,
            user_message="Hello",
            assistant_message="Hi",
        )

        await adapter.add(conv_id, turn)
        await adapter.clear(conv_id)
        history = await adapter.get_history(conv_id)

        assert len(history) == 0
