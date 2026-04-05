"""
Configuration schema for Bhodi platform.

All runtime configuration is driven through these Pydantic models.
No hardcoded values for models, temperatures, chunk sizes, or paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingConfig(BaseModel):
    """
    Embedding provider configuration.

    Provider-specific settings go in `extra`.
    """

    provider: str = Field(
        description="Embedding provider name (e.g., 'openai', 'local', 'mock')"
    )
    model: str | None = Field(
        default=None, description="Model name (provider-specific)"
    )
    dimensions: int | None = Field(
        default=None, description="Embedding dimensions (provider-specific)"
    )
    batch_size: int = Field(
        default=100, ge=1, le=1000, description="Batch size for embedding generation"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config"
    )


class VectorStoreConfig(BaseModel):
    """
    Vector store provider configuration.
    """

    provider: str = Field(
        description="Vector store provider name (e.g., 'chroma', 'qdrant', 'in_memory')"
    )
    persist_directory: Path | None = Field(
        default=None, description="Directory for persistent storage"
    )
    collection_name: str = Field(default="bhodi", description="Collection name")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config"
    )


class LLMConfig(BaseModel):
    """
    LLM provider configuration for generation.
    """

    provider: str = Field(
        description="LLM provider name (e.g., 'openai', 'anthropic', 'ollama', 'mock')"
    )
    model: str | None = Field(
        default=None, description="Model name (provider-specific)"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens to generate"
    )
    context_window: int | None = Field(
        default=None, ge=1, description="Context window size (provider-specific)"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config"
    )


class ChunkerConfig(BaseModel):
    """
    Text chunking configuration.
    """

    provider: str = Field(
        description="Chunking strategy (e.g., 'fixed_size', 'recursive', 'semantic')"
    )
    chunk_size: int | None = Field(
        default=None, ge=1, description="Target chunk size (model/provider-dependent)"
    )
    overlap: int | None = Field(
        default=None, ge=0, description="Overlap between chunks"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config"
    )


class DocumentParserConfig(BaseModel):
    """
    Document parsing configuration.
    """

    provider: str = Field(description="Parser provider (e.g., 'pypdf', 'unstructured')")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config"
    )


class ConversationConfig(BaseModel):
    """
    Conversation memory configuration.
    """

    provider: str = Field(
        description="Memory provider (e.g., 'volatile', 'persistent')"
    )
    max_history: int | None = Field(
        default=None, ge=1, description="Maximum turns to retain per conversation"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config"
    )


class TelemetryConfig(BaseModel):
    """
    OpenTelemetry configuration.
    """

    enabled: bool = Field(default=True, description="Enable telemetry spans")
    service_name: str = Field(default="bhodi", description="Service name for traces")
    exporter: str = Field(
        default="console", description="Exporter type ('console', 'otlp', 'none')"
    )
    otlp_endpoint: str | None = Field(
        default=None, description="OTLP collector endpoint"
    )


class BhodiConfig(BaseModel):
    """
    Root configuration for Bhodi platform.

    All values are optional with sensible defaults to allow partial overrides.
    """

    parser: DocumentParserConfig = Field(
        default_factory=lambda: DocumentParserConfig(provider="pypdf")
    )
    chunker: ChunkerConfig = Field(
        default_factory=lambda: ChunkerConfig(provider="recursive")
    )
    embedding: EmbeddingConfig = Field(
        default_factory=lambda: EmbeddingConfig(provider="openai")
    )
    vector_store: VectorStoreConfig = Field(
        default_factory=lambda: VectorStoreConfig(provider="chroma")
    )
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(provider="openai"))
    conversation: ConversationConfig = Field(
        default_factory=lambda: ConversationConfig(provider="volatile")
    )
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    model_config = ConfigDict(
        extra="ignore",  # Allow extra fields in config files without failing
    )
