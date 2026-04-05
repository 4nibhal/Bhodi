from __future__ import annotations

from bhodi_platform.evaluation.thresholds import (
    GroundingThresholds,
    RetrievalThresholds,
)

DEFAULT_RETRIEVAL_THRESHOLDS = RetrievalThresholds(
    hit_rate_min=0.5,
    pollution_rate_max=0.0,
    memory_leak_rate_max=0.0,
)

DEFAULT_GROUNDING_THRESHOLDS = GroundingThresholds(
    supported_fact_recall_min=0.7,
    unsupported_claim_rate_max=0.0,
)
