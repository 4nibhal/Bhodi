from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from bhodi_platform.retrieval.settings import (
    DEFAULT_EMBEDDINGS_DEVICE,
    DEFAULT_EMBEDDINGS_MODEL,
)


DEFAULT_LOCAL_MODEL = "models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
DEFAULT_TOKENIZER_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct"
DEFAULT_SUMMARIZER_MODEL = "facebook/bart-large-cnn"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOKENIZERS_PARALLELISM = "false"


@dataclass(frozen=True, slots=True)
class AnsweringSettings:
    local_model_path: Path
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    embeddings_model: str = DEFAULT_EMBEDDINGS_MODEL
    embeddings_device: str = DEFAULT_EMBEDDINGS_DEVICE
    summarizer_model: str = DEFAULT_SUMMARIZER_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    tokenizers_parallelism: str = DEFAULT_TOKENIZERS_PARALLELISM
    llm_temperature: float = 0.1
    llm_context_window: int = 3000
    llm_gpu_layers: int = -1
    llm_batch_size: int = 50
    llm_max_tokens: int = 3000
    llm_top_p: float = 0.9
    llm_verbose: bool = False
    volatile_retriever_k: int = 3
    context_token_limit: int = 2000
    context_char_limit: int = 1000
    prompt_summary_token_limit: int = 1200
    document_summary_token_limit: int = 300
    raw_summary_char_limit: int = 2500
    summarizer_max_length: int = 1500
    summarizer_min_length: int = 500
    reranker_max_length: int = 512
    tokenizer_max_length: int = 1024

    @classmethod
    def from_environment(cls, cwd: Path | None = None) -> "AnsweringSettings":
        base_directory = cwd or Path.cwd()
        local_model_path = Path(
            os.getenv("BHODI_LOCAL_MODEL", str(base_directory / DEFAULT_LOCAL_MODEL))
        )
        return cls(
            local_model_path=local_model_path,
            tokenizer_model=os.getenv("BHODI_TOKENIZER_MODEL", DEFAULT_TOKENIZER_MODEL),
            embeddings_model=os.getenv(
                "BHODI_EMBEDDINGS_MODEL", DEFAULT_EMBEDDINGS_MODEL
            ),
            embeddings_device=os.getenv(
                "BHODI_EMBEDDINGS_DEVICE", DEFAULT_EMBEDDINGS_DEVICE
            ),
            summarizer_model=os.getenv(
                "BHODI_SUMMARIZER_MODEL", DEFAULT_SUMMARIZER_MODEL
            ),
            reranker_model=os.getenv("BHODI_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
            tokenizers_parallelism=os.getenv(
                "TOKENIZERS_PARALLELISM", DEFAULT_TOKENIZERS_PARALLELISM
            ),
        )
