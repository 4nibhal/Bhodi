---
scope: "tests/"
type: "rules"
role: "Quality Gates"
parent: "/"
priority: high
metadata:
  system: "aiwf"
  suites: ["unit", "integration", "contract", "evals"]
---

# Rules: Tests

## Context & Responsibility
This scope governs automated verification for Bhodi product code and migration work.

## Operational Standards
- Add characterization coverage before changing legacy behavior.
- Keep tests deterministic, hermetic where possible, and explicit about fixtures.
- Separate unit, integration, contract, and retrieval-eval concerns.
- Retrieval or answer-quality changes require eval coverage or updated thresholds.
- CI should be able to run baseline tests through `uv` without requiring local ad hoc setup.

## Capability Graph
- @skill/legacy-modernization
- @skill/rag-quality
- @skill/python-release-engineering

## Delegation & Boundaries
- Inheritance: Inherits from @rules/root.
- Focus: Verification only; test helpers must not become hidden production dependencies.
