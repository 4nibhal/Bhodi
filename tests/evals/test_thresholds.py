from datetime import datetime, timezone
from unittest import TestCase

from bhodi_platform.evaluation.thresholds import (
    EvaluationBudget,
    GroundingThresholds,
    RetrievalThresholds,
)


class RetrievalThresholdsTest(TestCase):
    def test_creation_with_required_fields(self) -> None:
        thresholds = RetrievalThresholds(
            hit_rate_min=0.6,
            pollution_rate_max=0.1,
            memory_leak_rate_max=0.05,
        )
        self.assertEqual(thresholds.hit_rate_min, 0.6)
        self.assertEqual(thresholds.pollution_rate_max, 0.1)
        self.assertEqual(thresholds.memory_leak_rate_max, 0.05)

    def test_immutable(self) -> None:
        thresholds = RetrievalThresholds(0.5, 0.0, 0.0)
        with self.assertRaises(AttributeError):
            thresholds.hit_rate_min = 0.7


class GroundingThresholdsTest(TestCase):
    def test_creation_with_required_fields(self) -> None:
        thresholds = GroundingThresholds(
            supported_fact_recall_min=0.8,
            unsupported_claim_rate_max=0.05,
        )
        self.assertEqual(thresholds.supported_fact_recall_min, 0.8)
        self.assertEqual(thresholds.unsupported_claim_rate_max, 0.05)

    def test_immutable(self) -> None:
        thresholds = GroundingThresholds(0.7, 0.0)
        with self.assertRaises(AttributeError):
            thresholds.supported_fact_recall_min = 0.9


class EvaluationBudgetTest(TestCase):
    def test_create_with_current_timestamp(self) -> None:
        retrieval = RetrievalThresholds(0.5, 0.0, 0.0)
        grounding = GroundingThresholds(0.7, 0.0)
        budget = EvaluationBudget.create(
            retrieval=retrieval,
            grounding=grounding,
            run_id="test-run-001",
            environment="ci",
        )
        self.assertEqual(budget.run_id, "test-run-001")
        self.assertEqual(budget.environment, "ci")
        self.assertIsNotNone(budget.timestamp)
        self.assertEqual(budget.retrieval, retrieval)
        self.assertEqual(budget.grounding, grounding)

    def test_create_manual_timestamp(self) -> None:
        ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        retrieval = RetrievalThresholds(0.5, 0.0, 0.0)
        grounding = GroundingThresholds(0.7, 0.0)
        budget = EvaluationBudget(
            retrieval=retrieval,
            grounding=grounding,
            run_id="manual-run",
            timestamp=ts,
            environment="local",
        )
        self.assertEqual(budget.timestamp, ts)

    def test_to_dict(self) -> None:
        retrieval = RetrievalThresholds(0.5, 0.0, 0.0)
        grounding = GroundingThresholds(0.7, 0.0)
        budget = EvaluationBudget.create(
            retrieval=retrieval,
            grounding=grounding,
            run_id="dict-test",
            environment="test",
        )
        d = budget.to_dict()
        self.assertEqual(d["run_id"], "dict-test")
        self.assertEqual(d["environment"], "test")
        self.assertEqual(d["retrieval"]["hit_rate_min"], 0.5)
        self.assertEqual(d["grounding"]["supported_fact_recall_min"], 0.7)
