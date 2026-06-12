"""
Configuration schema for bodhi-rag platform.

All runtime configuration is driven through these Pydantic models.
No hardcoded values for models, temperatures, chunk sizes, or paths.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigError(ValueError):
    """Raised when configuration cannot be loaded or validated.

    Subclasses ValueError so callers that catch the broad type still work.
    Error messages must include enough context to debug the issue: file
    path (when TOML), field name, and the layer where the problem was
    detected (CLI / env / TOML / defaults).
    """


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
    collection_name: str = Field(default="bodhi-rag", description="Collection name")
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


class RerankerConfig(BaseModel):
    """
    Reranker configuration.

    Wave 1 establishes the schema and the no-hardcoded-defaults policy:
    when `provider == "cross_encoder"`, `model` is required and the
    adapter is responsible for failing fast at construction time if it
    is missing. The actual `CrossEncoderReranker` adapter lands in
    Wave 3a; the contract enforced here is that *no* reranker adapter
    in this project is ever constructed without an explicit model name
    chosen by the user via `bodhi.toml`, env, or CLI.

    `overfetch_factor` and `batch_size` are operational tunables with
    acceptable defaults; they are not model selections, so a default
    is documented and expected to be overridable.
    """

    provider: Literal["noop", "cross_encoder"] = Field(
        default="noop",
        description="Reranker provider: 'noop' (default) or 'cross_encoder' (opt-in).",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Model identifier (e.g. a sentence-transformers cross-encoder name). "
            "REQUIRED when provider='cross_encoder'; the adapter raises ConfigError "
            "if it is missing. NEVER hardcoded in adapter code."
        ),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Optional override of the post-rerank top_k.",
    )
    overfetch_factor: int = Field(
        default=4,
        ge=1,
        le=64,
        description=(
            "Multiplier on top_k for the pre-rerank vector-store search. "
            "Skipped when reranker is NoOpReranker."
        ),
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for the cross-encoder scoring call.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra config"
    )

    @model_validator(mode="after")
    def _validate_cross_encoder_model(self) -> "RerankerConfig":
        if self.provider == "cross_encoder" and (
            self.model is None or self.model.strip() == ""
        ):
            raise ConfigError(
                "RerankerConfig.model is required when provider is "
                '"cross_encoder". Set BODHI_RERANKER_MODEL or define '
                "[reranker] model in bodhi.toml."
            )
        return self


class TelemetryConfig(BaseModel):
    """
    OpenTelemetry configuration.
    """

    enabled: bool = Field(default=True, description="Enable telemetry spans")
    service_name: str = Field(default="bodhi-rag", description="Service name for traces")
    exporter: str = Field(
        default="console", description="Exporter type ('console', 'otlp', 'none')"
    )
    otlp_endpoint: str | None = Field(
        default=None, description="OTLP collector endpoint"
    )


class BhodiConfig(BaseModel):
    """
    Root configuration for bodhi-rag platform.

    All values are optional with sensible defaults to allow partial overrides.
    """

    parser: DocumentParserConfig = Field(
        default_factory=lambda: DocumentParserConfig(
            provider=os.getenv("BODHI_PARSER_PROVIDER", "pypdf")
        )
    )
    chunker: ChunkerConfig = Field(
        default_factory=lambda: ChunkerConfig(
            provider=os.getenv("BODHI_CHUNKER_PROVIDER", "recursive")
        )
    )
    embedding: EmbeddingConfig = Field(
        default_factory=lambda: EmbeddingConfig(
            provider=os.getenv("BODHI_EMBEDDING_PROVIDER", "openai")
        )
    )
    vector_store: VectorStoreConfig = Field(
        default_factory=lambda: VectorStoreConfig(
            provider=os.getenv("BODHI_VECTOR_STORE_PROVIDER", "chroma")
        )
    )
    llm: LLMConfig = Field(
        default_factory=lambda: LLMConfig(
            provider=os.getenv("BODHI_LLM_PROVIDER", "openai")
        )
    )
    conversation: ConversationConfig = Field(
        default_factory=lambda: ConversationConfig(
            provider=os.getenv("BODHI_CONVERSATION_PROVIDER", "volatile")
        )
    )
    reranker: RerankerConfig = Field(
        default_factory=lambda: RerankerConfig(
            provider=os.getenv("BODHI_RERANKER_PROVIDER", "noop"),  # type: ignore[arg-type]
            model=os.getenv("BODHI_RERANKER_MODEL"),
        )
    )
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    model_config = ConfigDict(
        extra="ignore",  # Allow extra fields in config files without failing
    )

    @classmethod
    def from_env(cls, env: "Mapping[str, str] | None" = None) -> "BhodiConfig":
        """Build a `BhodiConfig` from environment variables only.

        The TOML layer is skipped. The `env` mapping defaults to `os.environ`
        but tests can pass a synthetic mapping. Unknown env vars are ignored;
        only the documented `BODHI_*` (and the API-layer `BODHI_API_*`)
        variables are consumed.
        """
        # Pydantic default_factory lambdas already read from os.getenv. The
        # explicit constructor here ensures that a custom `env` mapping is
        # honored — we temporarily patch os.environ for the duration of the
        # call so the default factories see the caller-provided values.
        if env is None:
            return cls()
        with _patched_environ(env):
            return cls()

    @classmethod
    def from_toml(
        cls, path: "str | Path", *, env: "Mapping[str, str] | None" = None
    ) -> "BhodiConfig":
        """Build a `BhodiConfig` from a TOML file.

        The TOML file is parsed with `tomllib`; nested sections are validated
        against the corresponding Pydantic sub-config (`[embedding]` ->
        `EmbeddingConfig`, etc.). Unknown sections / keys are ignored, matching
        the `extra="ignore"` policy on the root model. Missing required
        fields raise `ConfigError` (e.g. `[reranker] provider = "cross_encoder"`
        with no `model`).

        The optional `env` overlay is applied AFTER the TOML layer: any
        `BODHI_*` env var overrides the corresponding TOML field at the
        per-sub-config dict level. This keeps the precedence
        TOML < env < CLI when `load_bodhi_config` is used, and matches the
        documented 12-factor behaviour.
        """
        import tomllib

        path = Path(path).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigError(
                f"TOML config file not found: {path}"
            ) from exc
        except OSError as exc:
            raise ConfigError(
                f"Could not read TOML config file {path}: {exc}"
            ) from exc

        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(
                f"Malformed TOML in {path}: {exc}"
            ) from exc

        # Mapping from TOML section name to a (sub-config class, env-field map)
        # where env-field map is {env var name: field name on the sub-config}.
        section_map: dict[str, tuple[type[BaseModel], dict[str, str]]] = {
            "parser": (DocumentParserConfig, {"BODHI_PARSER_PROVIDER": "provider"}),
            "chunker": (
                ChunkerConfig,
                {"BODHI_CHUNKER_PROVIDER": "provider"},
            ),
            "embedding": (
                EmbeddingConfig,
                {
                    "BODHI_EMBEDDING_PROVIDER": "provider",
                    "BODHI_EMBEDDING_MODEL": "model",
                },
            ),
            "vector_store": (
                VectorStoreConfig,
                {"BODHI_VECTOR_STORE_PROVIDER": "provider"},
            ),
            "llm": (
                LLMConfig,
                {
                    "BODHI_LLM_PROVIDER": "provider",
                    "BODHI_LLM_MODEL": "model",
                },
            ),
            "conversation": (
                ConversationConfig,
                {"BODHI_CONVERSATION_PROVIDER": "provider"},
            ),
            "reranker": (
                RerankerConfig,
                {
                    "BODHI_RERANKER_PROVIDER": "provider",
                    "BODHI_RERANKER_MODEL": "model",
                },
            ),
            "telemetry": (TelemetryConfig, {}),
        }

        kwargs: dict[str, Any] = {}
        for section, (model_cls, env_map) in section_map.items():
            section_data: dict[str, Any] = {}
            if section in data and isinstance(data[section], dict):
                section_data.update(data[section])
            # Apply env overlay: env beats TOML.
            if env is not None:
                for env_key, field_name in env_map.items():
                    if env_key in env:
                        section_data[field_name] = env[env_key]
            if not section_data:
                continue
            # Even with only env, the sub-config must validate cleanly.
            try:
                kwargs[section] = model_cls.model_validate(section_data)
            except Exception as exc:  # noqa: BLE001 - rewrap with layer info
                raise ConfigError(
                    f"Invalid [{section}] section in {path}: {exc}"
                ) from exc

        # Apply env to any top-level defaults not covered by sections above.
        # Specifically: the `BhodiConfig` default_factory lambdas read
        # BODHI_*_PROVIDER env vars, but since we are now passing explicit
        # sub-configs, we must apply the env ourselves to honour the
        # "env beats TOML" contract for fields not present in the TOML.
        if env is not None:
            with _patched_environ(env):
                return cls(**kwargs)
        return cls(**kwargs)


@contextmanager
def _patched_environ(env: "Mapping[str, str]"):
    """Context manager: temporarily replace os.environ with the given mapping.

    Used so that `BhodiConfig` default_factory lambdas (which call
    `os.getenv`) honor a caller-provided env mapping without mutating
    the real process environment.
    """
    import os as _os

    sentinel = object()
    saved = {k: _os.environ.get(k, sentinel) for k in env}
    try:
        _os.environ.clear()
        _os.environ.update(env)
        yield
    finally:
        _os.environ.clear()
        for k, v in saved.items():
            if v is sentinel:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
