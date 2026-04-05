from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PERSIST_DIRECTORY_NAME = "chroma_db"


@dataclass(frozen=True, slots=True)
class IndexingSettings:
    persist_directory: Path
    retriever_k: int = 3
    chunk_size: int = 1000
    chunk_overlap: int = 200

    @classmethod
    def from_environment(cls, cwd: Path | None = None) -> "IndexingSettings":
        base_directory = cwd or Path.cwd()
        persist_directory = Path(
            os.getenv(
                "BHODI_INDEX_PERSIST_DIRECTORY",
                str(base_directory / DEFAULT_PERSIST_DIRECTORY_NAME),
            )
        )
        return cls(persist_directory=persist_directory)
