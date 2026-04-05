from __future__ import annotations

from dataclasses import dataclass, field

from bhodi_platform.evaluation.models import (
    GroundingCase,
    RetrievedArtifact,
    RetrievalCase,
)


@dataclass(frozen=True, slots=True)
class RetrievalCaseScore:
    query_id: str
    hit: bool
    polluted: bool
    matched_document_ids: tuple[str, ...] = field(default_factory=tuple)
    forbidden_origins: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RetrievalSuiteScore:
    total_cases: int
    hit_cases: int
    pollution_free_cases: int
    case_scores: tuple[RetrievalCaseScore, ...]

    @property
    def hit_rate(self) -> float:
        """Return the hit rate: hit_cases / total_cases."""
        if self.total_cases == 0:
            return 0.0
        return self.hit_cases / self.total_cases

    @property
    def pollution_free_rate(self) -> float:
        """Return the pollution-free rate: pollution_free_cases / total_cases."""
        if self.total_cases == 0:
            return 0.0
        return self.pollution_free_cases / self.total_cases


@dataclass(frozen=True, slots=True)
class GroundingCaseScore:
    query_id: str
    grounded: bool
    supported_sources: tuple[str, ...] = field(default_factory=tuple)
    missing_sources: tuple[str, ...] = field(default_factory=tuple)
    unsupported_citations: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class GroundingSuiteScore:
    total_cases: int
    grounded_cases: int
    case_scores: tuple[GroundingCaseScore, ...]

    @property
    def grounded_rate(self) -> float:
        """Return the grounded rate: grounded_cases / total_cases."""
        if self.total_cases == 0:
            return 0.0
        return self.grounded_cases / self.total_cases

    @property
    def supported_fact_recall(self) -> float:
        """Return the supported fact recall: grounded_cases / total_cases.

        This measures the fraction of cases where all required sources are supported.
        """
        return self.grounded_rate

    @property
    def unsupported_claim_rate(self) -> float:
        """Return the unsupported claim rate per case on average."""
        if self.total_cases == 0:
            return 0.0
        total_unsupported = sum(
            len(score.unsupported_citations) for score in self.case_scores
        )
        return total_unsupported / self.total_cases


def score_retrieval_case(
    case: RetrievalCase,
    retrieved_artifacts: list[RetrievedArtifact],
) -> RetrievalCaseScore:
    retrieved_ids = {artifact.document_id for artifact in retrieved_artifacts}
    matched_document_ids = tuple(
        document_id
        for document_id in case.expected_document_ids
        if document_id in retrieved_ids
    )
    forbidden_origins = tuple(
        origin
        for origin in case.forbidden_origins
        if any(artifact.origin == origin for artifact in retrieved_artifacts)
    )
    return RetrievalCaseScore(
        query_id=case.query_id,
        hit=len(matched_document_ids) == len(case.expected_document_ids),
        polluted=len(forbidden_origins) > 0,
        matched_document_ids=matched_document_ids,
        forbidden_origins=forbidden_origins,
    )


def score_retrieval_suite(
    cases: tuple[RetrievalCase, ...],
    results_by_query: dict[str, list[RetrievedArtifact]],
) -> RetrievalSuiteScore:
    case_scores = tuple(
        score_retrieval_case(case, results_by_query.get(case.query_id, []))
        for case in cases
    )
    return RetrievalSuiteScore(
        total_cases=len(case_scores),
        hit_cases=sum(score.hit for score in case_scores),
        pollution_free_cases=sum(not score.polluted for score in case_scores),
        case_scores=case_scores,
    )


def score_grounding_case(
    case: GroundingCase,
    supporting_artifacts: list[RetrievedArtifact],
    cited_sources: tuple[str, ...],
) -> GroundingCaseScore:
    supporting_sources = {
        artifact.source
        for artifact in supporting_artifacts
        if artifact.source is not None
    }
    missing_sources = tuple(
        source for source in case.required_sources if source not in supporting_sources
    )
    unsupported_citations = tuple(
        source for source in cited_sources if source not in supporting_sources
    )
    return GroundingCaseScore(
        query_id=case.query_id,
        grounded=not missing_sources and not unsupported_citations,
        supported_sources=tuple(sorted(supporting_sources)),
        missing_sources=missing_sources,
        unsupported_citations=unsupported_citations,
    )


def score_grounding_suite(
    cases: tuple[GroundingCase, ...],
    supporting_by_query: dict[str, list[RetrievedArtifact]],
    citations_by_query: dict[str, tuple[str, ...]],
) -> GroundingSuiteScore:
    case_scores = tuple(
        score_grounding_case(
            case,
            supporting_by_query.get(case.query_id, []),
            citations_by_query.get(case.query_id, ()),
        )
        for case in cases
    )
    return GroundingSuiteScore(
        total_cases=len(case_scores),
        grounded_cases=sum(score.grounded for score in case_scores),
        case_scores=case_scores,
    )
