from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PERSIST_DIRECTORY_NAME = "chroma_conversations"
DEFAULT_COLLECTION_NAME = "conversation_memory"


@dataclass(frozen=True, slots=True)
class ConversationSettings:
    persist_directory: Path
    collection_name: str = DEFAULT_COLLECTION_NAME
    retriever_k: int = 3

    @classmethod
    def from_environment(cls, cwd: Path | None = None) -> "ConversationSettings":
        base_directory = cwd or Path.cwd()
        persist_directory = Path(
            os.getenv(
                "BHODI_CONVERSATION_PERSIST_DIRECTORY",
                str(base_directory / DEFAULT_PERSIST_DIRECTORY_NAME),
            )
        )
        retriever_k = int(os.getenv("BHODI_CONVERSATION_RETRIEVER_K", "3"))
        collection_name = os.getenv(
            "BHODI_CONVERSATION_COLLECTION_NAME",
            DEFAULT_COLLECTION_NAME,
        )
        return cls(
            persist_directory=persist_directory,
            collection_name=collection_name,
            retriever_k=retriever_k,
        )
