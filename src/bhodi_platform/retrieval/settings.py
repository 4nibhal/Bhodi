from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDINGS_DEVICE = "cuda"


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    model: str = DEFAULT_EMBEDDINGS_MODEL
    device: str = DEFAULT_EMBEDDINGS_DEVICE

    @classmethod
    def from_environment(cls) -> "EmbeddingSettings":
        return cls(
            model=os.getenv("BHODI_EMBEDDINGS_MODEL", DEFAULT_EMBEDDINGS_MODEL),
            device=os.getenv("BHODI_EMBEDDINGS_DEVICE", DEFAULT_EMBEDDINGS_DEVICE),
        )
