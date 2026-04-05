from __future__ import annotations

from bhodi_platform.evaluation.scoring import (
    GroundingSuiteScore,
    RetrievalSuiteScore,
)
from bhodi_platform.evaluation.thresholds import (
    EvaluationBudget,
    GroundingThresholds,
    RetrievalThresholds,
)


class BudgetValidator:
    """Validates suite scores against threshold budgets."""

    def validate_retrieval(
        self, suite_score: RetrievalSuiteScore, thresholds: RetrievalThresholds
    ) -> tuple[bool, list[str]]:
        """Validate retrieval suite score against thresholds.

        Returns:
            A tuple of (passes, failure_reasons).
        """
        failures: list[str] = []

        hit_rate = suite_score.hit_rate
        if hit_rate < thresholds.hit_rate_min:
            failures.append(
                f"hit_rate {hit_rate:.3f} < threshold {thresholds.hit_rate_min:.3f}"
            )

        pollution_rate = 1.0 - suite_score.pollution_free_rate
        if pollution_rate > thresholds.pollution_rate_max:
            failures.append(
                f"pollution_rate {pollution_rate:.3f} > threshold {thresholds.pollution_rate_max:.3f}"
            )

        memory_leak_rate = pollution_rate  # Using pollution as proxy for memory_leak
        if memory_leak_rate > thresholds.memory_leak_rate_max:
            failures.append(
                f"memory_leak_rate {memory_leak_rate:.3f} > threshold {thresholds.memory_leak_rate_max:.3f}"
            )

        return (len(failures) == 0, failures)

    def validate_grounding(
        self, suite_score: GroundingSuiteScore, thresholds: GroundingThresholds
    ) -> tuple[bool, list[str]]:
        """Validate grounding suite score against thresholds.

        Returns:
            A tuple of (passes, failure_reasons).
        """
        failures: list[str] = []

        if suite_score.supported_fact_recall < thresholds.supported_fact_recall_min:
            failures.append(
                f"supported_fact_recall {suite_score.supported_fact_recall:.3f} < "
                f"threshold {thresholds.supported_fact_recall_min:.3f}"
            )

        if suite_score.unsupported_claim_rate > thresholds.unsupported_claim_rate_max:
            failures.append(
                f"unsupported_claim_rate {suite_score.unsupported_claim_rate:.3f} > "
                f"threshold {thresholds.unsupported_claim_rate_max:.3f}"
            )

        return (len(failures) == 0, failures)

    def validate_budget(
        self,
        budget: EvaluationBudget,
        suite_scores: tuple[RetrievalSuiteScore, GroundingSuiteScore],
    ) -> tuple[bool, dict[str, bool | list[str]]]:
        """Validate both retrieval and grounding suite scores against a budget.

        Args:
            budget: The evaluation budget with thresholds.
            suite_scores: Tuple of (retrieval_score, grounding_score).

        Returns:
            A tuple of (overall_passes, details) where details contains
            per-component pass/fail status and failure reasons.
        """
        retrieval_score, grounding_score = suite_scores

        retrieval_passes, retrieval_failures = self.validate_retrieval(
            retrieval_score, budget.retrieval
        )
        grounding_passes, grounding_failures = self.validate_grounding(
            grounding_score, budget.grounding
        )

        details: dict[str, bool | list[str]] = {
            "retrieval_passes": retrieval_passes,
            "retrieval_failures": retrieval_failures,
            "grounding_passes": grounding_passes,
            "grounding_failures": grounding_failures,
        }

        return (retrieval_passes and grounding_passes, details)
