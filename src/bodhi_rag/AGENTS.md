---
scope: "src/bodhi_rag/"
type: "rules"
role: "Backend Platform Rules for bodhi-rag"
parent: "src/"
priority: high
metadata:
  system: "aiwf"
  bounded_contexts: ["domain", "application", "ports", "infrastructure", "interfaces", "evaluation"]
---

# Rules: Backend Platform

## Context & Responsibility
This scope defines the target backend architecture for bodhi-rag. It is the preferred home for all modernized document ingestion, retrieval, generation, and evaluation capabilities.

## Architectural Contract
- `domain/` contains pure entities, value objects, and business rules.
- `application/` contains use cases and orchestration.
- `ports/` defines abstract dependencies.
- `infrastructure/` implements adapters such as Chroma, filesystem, Hugging Face, and llama.cpp.
- `interfaces/` exposes transports and UX adapters.
- `evaluation/` contains quality fixtures, metrics, and regression gates.

## Operational Standards
- Backend modules must not import TUI-specific code.
- Retrieval and generation flows must preserve source metadata and support citations.
- Any RAG change must define how it will be evaluated.
- Services must be composable for API and CLI consumers.

## Capability Graph
- @skill/legacy-modernization
- @skill/rag-quality
- @skill/python-release-engineering

## Delegation & Boundaries

### Nested Rules

| Rule Scope | Location |
| :--- | :--- |
| `src/bodhi_rag/application/` | [src/bodhi_rag/application/AGENTS.md](/src/bodhi_rag/application/AGENTS.md) |
| `src/bodhi_rag/domain/` | [src/bodhi_rag/domain/AGENTS.md](/src/bodhi_rag/domain/AGENTS.md) |
| `src/bodhi_rag/infrastructure/` | [src/bodhi_rag/infrastructure/AGENTS.md](/src/bodhi_rag/infrastructure/AGENTS.md) |
| `src/bodhi_rag/interfaces/` | [src/bodhi_rag/interfaces/AGENTS.md](/src/bodhi_rag/interfaces/AGENTS.md) |
| `src/bodhi_rag/ports/` | [src/bodhi_rag/ports/AGENTS.md](/src/bodhi_rag/ports/AGENTS.md) |
