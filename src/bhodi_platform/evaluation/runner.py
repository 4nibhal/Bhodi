from __future__ import annotations

from bhodi_platform.evaluation.budget import BudgetValidator
from bhodi_platform.evaluation.loader import load_fixture
from bhodi_platform.evaluation.models import RetrievedArtifact
from bhodi_platform.evaluation.scoring import (
    GroundingSuiteScore,
    RetrievalSuiteScore,
    score_grounding_suite,
    score_retrieval_suite,
)
from bhodi_platform.evaluation.thresholds import (
    GroundingThresholds,
    RetrievalThresholds,
)


class EvaluationRunner:
    """Runs retrieval and grounding evaluations against fixtures and thresholds."""

    def __init__(self) -> None:
        self.validator = BudgetValidator()

    def run_retrieval_eval(
        self,
        fixture_path: str,
        thresholds: RetrievalThresholds,
        artifacts: dict[str, list[RetrievedArtifact]],
    ) -> tuple[RetrievalSuiteScore, bool, list[str]]:
        """Run retrieval evaluation.

        Args:
            fixture_path: Name of the fixture file to load.
            thresholds: Thresholds to validate against.
            artifacts: Map of query_id to retrieved artifacts.

        Returns:
            Tuple of (score, passes, failure_reasons).
        """
        fixture = load_fixture(fixture_path)
        score = score_retrieval_suite(fixture.retrieval_cases, artifacts)
        passes, failures = self.validator.validate_retrieval(score, thresholds)
        return (score, passes, failures)

    def run_grounding_eval(
        self,
        fixture_path: str,
        thresholds: GroundingThresholds,
        supporting_artifacts: dict[str, list[RetrievedArtifact]],
        citations: dict[str, tuple[str, ...]],
    ) -> tuple[GroundingSuiteScore, bool, list[str]]:
        """Run grounding evaluation.

        Args:
            fixture_path: Name of the fixture file to load.
            thresholds: Thresholds to validate against.
            supporting_artifacts: Map of query_id to supporting artifacts.
            citations: Map of query_id to cited sources.

        Returns:
            Tuple of (score, passes, failure_reasons).
        """
        fixture = load_fixture(fixture_path)
        score = score_grounding_suite(
            fixture.grounding_cases, supporting_artifacts, citations
        )
        passes, failures = self.validator.validate_grounding(score, thresholds)
        return (score, passes, failures)
