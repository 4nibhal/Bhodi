---
scope: "src/bhodi_platform/"
type: "rules"
role: "Backend Platform Rules"
parent: "src/"
priority: high
metadata:
  system: "aiwf"
  bounded_contexts: ["domain", "application", "ports", "infrastructure", "interfaces", "evaluation"]
---

# Rules: Backend Platform

## Context & Responsibility
This scope defines the target backend architecture for Bhodi. It is the preferred home for all modernized document ingestion, retrieval, generation, and evaluation capabilities.

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
- Services must be composable for API, CLI, worker, and TUI consumers.

## Capability Graph
- @skill/legacy-modernization
- @skill/rag-quality
- @skill/python-release-engineering

## Delegation & Boundaries

### Nested Rules

| Rule Scope | Location |
| :--- | :--- |
| `src/bhodi_platform/application/` | [src/bhodi_platform/application/AGENTS.md](/src/bhodi_platform/application/AGENTS.md) |
| `src/bhodi_platform/domain/` | [src/bhodi_platform/domain/AGENTS.md](/src/bhodi_platform/domain/AGENTS.md) |
| `src/bhodi_platform/infrastructure/` | [src/bhodi_platform/infrastructure/AGENTS.md](/src/bhodi_platform/infrastructure/AGENTS.md) |
| `src/bhodi_platform/interfaces/` | [src/bhodi_platform/interfaces/AGENTS.md](/src/bhodi_platform/interfaces/AGENTS.md) |
| `src/bhodi_platform/interfaces/tui/` | [src/bhodi_platform/interfaces/tui/AGENTS.md](/src/bhodi_platform/interfaces/tui/AGENTS.md) |
| `src/bhodi_platform/ports/` | [src/bhodi_platform/ports/AGENTS.md](/src/bhodi_platform/ports/AGENTS.md) |
