from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from bhodi_platform.application import ConversationMessage


@dataclass(frozen=True, slots=True)
class WorkerRetryPolicy:
    max_attempts: int = 1
    retry_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_retry_delay_seconds: float = 60.0

    def next_delay(self, attempt: int) -> float:
        delay = self.retry_delay_seconds * (self.backoff_multiplier ** (attempt - 1))
        return min(delay, self.max_retry_delay_seconds)


@dataclass(frozen=True, slots=True)
class WorkerJobMetadata:
    attempt: int = 1
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class IndexDocumentsJob:
    job_id: str
    document_path: Path
    cwd: Path | None = None
    kind: Literal["index_documents"] = "index_documents"


@dataclass(frozen=True, slots=True)
class AnswerQueryJob:
    job_id: str
    user_input: str
    messages: tuple[ConversationMessage, ...] = field(default_factory=tuple)
    conversation_id: str | None = None
    kind: Literal["answer_query"] = "answer_query"


WorkerJob = IndexDocumentsJob | AnswerQueryJob


@dataclass(frozen=True, slots=True)
class IndexDocumentsJobResult:
    job_id: str
    indexed_fragments: int
    source_kind: str
    resolved_path: Path
    kind: Literal["index_documents"] = "index_documents"
    status: Literal["succeeded"] = "succeeded"


@dataclass(frozen=True, slots=True)
class AnswerQueryJobResult:
    job_id: str
    answer_text: str
    context: str = ""
    kind: Literal["answer_query"] = "answer_query"
    status: Literal["succeeded"] = "succeeded"


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    job_id: str
    kind: str
    error: str
    attempt: int = 1
    status: Literal["failed"] = "failed"


WorkerResult = IndexDocumentsJobResult | AnswerQueryJobResult | WorkerFailure
