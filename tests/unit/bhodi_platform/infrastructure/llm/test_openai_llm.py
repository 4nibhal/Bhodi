"""Unit tests for OpenAI LLM adapter."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from bhodi_platform.application.config import LLMConfig
from bhodi_platform.domain.exceptions import LLMError
from bhodi_platform.domain.entities import RetrievedDocument
from bhodi_platform.domain.value_objects import ChunkId, DocumentId
from bhodi_platform.infrastructure.llm.openai import OpenAILLMAdapter


class TestOpenAILLMAdapter:
    """Test suite for OpenAILLMAdapter."""

    @pytest.fixture
    def config(self):
        """Default LLM config for testing."""
        return LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            extra={"api_key": "FAKE_API_KEY_FOR_TESTS"},
        )

    @pytest.fixture
    def adapter(self, config):
        """Adapter instance for testing."""
        return OpenAILLMAdapter(config)

    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI chat completion response."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Generated response"
        return response

    @pytest.mark.asyncio
    async def test_generate_calls_openai_with_correct_params(
        self, adapter, mock_openai_response
    ):
        """generate() calls openai.AsyncClient.chat.completions.create with correct parameters."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        with patch(
            "openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            result = await adapter.generate(
                "Tell me a joke",
                temperature=0.5,
                max_tokens=100,
            )

        mock_client.chat.completions.create.assert_awaited_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Tell me a joke"}],
            temperature=0.5,
            max_tokens=100,
        )
        assert result == "Generated response"

    @pytest.mark.asyncio
    async def test_generate_with_context_builds_prompt_and_calls_generate(
        self, adapter, mock_openai_response
    ):
        """generate_with_context() builds prompt with context and calls generate."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        doc_id = DocumentId()
        chunk_id = ChunkId(document_id=doc_id, chunk_index=0)
        contexts = [
            RetrievedDocument(
                chunk_id=chunk_id,
                document_id=doc_id,
                text="Bhodi is a document processing platform.",
                score=0.95,
            )
        ]

        with patch(
            "openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            result = await adapter.generate_with_context(
                "What is Bhodi?",
                contexts,
                temperature=0.3,
            )

        mock_client.chat.completions.create.assert_awaited_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs

        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.3
        assert "Bhodi is a document processing platform" in call_kwargs["messages"][0]["content"]
        assert "What is Bhodi?" in call_kwargs["messages"][0]["content"]
        assert result == "Generated response"

    @pytest.mark.asyncio
    async def test_generate_wraps_openai_exception_in_llm_error(self, adapter):
        """generate() wraps OpenAI exceptions in LLMError."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API rate limit exceeded")
        )

        with patch(
            "openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            with pytest.raises(LLMError) as exc_info:
                await adapter.generate("Hello")

        assert "API rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.operation == "generate"

    @pytest.mark.asyncio
    async def test_generate_with_context_wraps_exception_in_llm_error(self, adapter):
        """generate_with_context() wraps exceptions in LLMError."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Connection timeout")
        )

        doc_id = DocumentId()
        chunk_id = ChunkId(document_id=doc_id, chunk_index=0)
        contexts = [
            RetrievedDocument(
                chunk_id=chunk_id,
                document_id=doc_id,
                text="Some context",
                score=0.9,
            )
        ]

        with patch(
            "openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            with pytest.raises(LLMError) as exc_info:
                await adapter.generate_with_context("Query?", contexts)

        assert "Connection timeout" in str(exc_info.value)

    def test_default_model_when_config_model_is_none(self):
        """Adapter uses default model when config model is not set."""
        config = LLMConfig(provider="openai")
        adapter = OpenAILLMAdapter(config)
        assert adapter._model == "gpt-4o-mini"

    def test_model_from_config(self):
        """Adapter uses model from config when provided."""
        config = LLMConfig(provider="openai", model="gpt-4o")
        adapter = OpenAILLMAdapter(config)
        assert adapter._model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate_uses_config_defaults(self, adapter, mock_openai_response):
        """generate() falls back to config defaults when kwargs not provided."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        with patch(
            "openai.AsyncOpenAI",
            return_value=mock_client,
        ):
            await adapter.generate("Hello")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == adapter._config.temperature
        assert call_kwargs["max_tokens"] == 2048
