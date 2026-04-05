---
scope: "src/bhodi_platform/interfaces/tui/"
type: "rules"
role: "Legacy TUI Adapter Rules"
parent: "src/bhodi_platform/"
priority: medium
metadata:
  system: "aiwf"
  status: "adapter-only"
---

# Rules: TUI Adapter

## Context & Responsibility
This scope covers the optional TUI interface for Bhodi. The TUI exists as a client adapter over backend services and must not own product orchestration.

## Operational Standards
- No direct model loading or vector store initialization in the TUI layer.
- No retrieval, indexing, or generation business rules in widgets or UI controllers.
- TUI changes should preserve parity with backend application contracts.
- Prefer thin presenters, request mappers, and response rendering.

## Capability Graph
- @skill/legacy-modernization

## Delegation & Boundaries
- Inheritance: Inherits from @rules/src/bhodi_platform.
- Focus: Adapter-only UI behavior; backend capability changes belong upstream.
