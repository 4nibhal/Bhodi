from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalThresholds:
    """Thresholds for retrieval evaluation."""

    hit_rate_min: float
    pollution_rate_max: float
    memory_leak_rate_max: float


@dataclass(frozen=True, slots=True)
class GroundingThresholds:
    """Thresholds for grounding evaluation."""

    supported_fact_recall_min: float
    unsupported_claim_rate_max: float


@dataclass(frozen=True, slots=True)
class EvaluationBudget:
    """Combined budget for retrieval and grounding evaluation with run metadata."""

    retrieval: RetrievalThresholds
    grounding: GroundingThresholds
    run_id: str
    timestamp: datetime
    environment: str

    @classmethod
    def create(
        cls,
        retrieval: RetrievalThresholds,
        grounding: GroundingThresholds,
        run_id: str,
        environment: str = "test",
    ) -> EvaluationBudget:
        """Create a budget with current UTC timestamp."""
        return cls(
            retrieval=retrieval,
            grounding=grounding,
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            environment=environment,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize budget to a dictionary."""
        return {
            "retrieval": {
                "hit_rate_min": self.retrieval.hit_rate_min,
                "pollution_rate_max": self.retrieval.pollution_rate_max,
                "memory_leak_rate_max": self.retrieval.memory_leak_rate_max,
            },
            "grounding": {
                "supported_fact_recall_min": self.grounding.supported_fact_recall_min,
                "unsupported_claim_rate_max": self.grounding.unsupported_claim_rate_max,
            },
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "environment": self.environment,
        }
