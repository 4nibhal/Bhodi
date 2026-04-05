from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from bhodi_platform.evaluation.models import (
    EvaluationFixture,
    GroundingCase,
    RetrievalCase,
)


def load_fixture(name: str = "retrieval_grounding_baseline.json") -> EvaluationFixture:
    raw_fixture = (
        files("bhodi_platform.evaluation.fixtures")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )
    payload = json.loads(raw_fixture)
    return _parse_fixture(payload)


def _parse_fixture(payload: dict[str, Any]) -> EvaluationFixture:
    name = payload.get("name")
    if not isinstance(name, str) or name == "":
        raise ValueError("Evaluation fixture must define a non-empty name.")

    raw_retrieval_cases = payload.get("retrieval_cases", [])
    raw_grounding_cases = payload.get("grounding_cases", [])
    if not isinstance(raw_retrieval_cases, list) or not isinstance(
        raw_grounding_cases, list
    ):
        raise ValueError("Fixture cases must be provided as lists.")

    retrieval_cases = tuple(
        RetrievalCase(
            query_id=_required_text(case, "query_id"),
            user_input=_required_text(case, "user_input"),
            expected_document_ids=_text_tuple(case.get("expected_document_ids", [])),
            forbidden_origins=_text_tuple(case.get("forbidden_origins", [])),
        )
        for case in _case_dicts(raw_retrieval_cases)
    )
    grounding_cases = tuple(
        GroundingCase(
            query_id=_required_text(case, "query_id"),
            required_sources=_text_tuple(case.get("required_sources", [])),
        )
        for case in _case_dicts(raw_grounding_cases)
    )
    return EvaluationFixture(
        name=name,
        retrieval_cases=retrieval_cases,
        grounding_cases=grounding_cases,
    )


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"Expected non-empty string for {key!r}.")
    return value


def _text_tuple(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list) or any(
        not isinstance(item, str) for item in values
    ):
        raise ValueError("Fixture list values must all be strings.")
    return tuple(values)


def _case_dicts(values: list[Any]) -> list[dict[str, Any]]:
    if any(not isinstance(item, dict) for item in values):
        raise ValueError("Fixture cases must be objects.")
    return values
