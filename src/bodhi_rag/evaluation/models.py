from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetrievedArtifact:
    document_id: str
    origin: str
    source: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    query_id: str
    user_input: str
    expected_document_ids: tuple[str, ...] = field(default_factory=tuple)
    forbidden_origins: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class GroundingCase:
    query_id: str
    required_sources: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EvaluationFixture:
    name: str
    retrieval_cases: tuple[RetrievalCase, ...] = field(default_factory=tuple)
    grounding_cases: tuple[GroundingCase, ...] = field(default_factory=tuple)
