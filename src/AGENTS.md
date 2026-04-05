---
scope: "src/"
type: "rules"
role: "Product Source Rules"
parent: "/"
priority: high
metadata:
  system: "aiwf"
  language: "python"
  architecture_style: "hexagonal"
---

# Rules: Product Source

## Context & Responsibility
This scope governs all shipped product code under `src/`. Bhodi is evolving from a legacy TUI-first codebase into a modular backend platform.

## Operational Standards
- Put new core logic under `src/bhodi_platform/`.
- Treat `src/bhodi_doc_analyzer/` and `src/indexer/` as legacy compatibility surfaces unless a task is explicitly modernizing them.
- Eliminate import-time side effects and global singleton ownership.
- Use typed settings, explicit factories, and dependency inversion boundaries.
- Keep filesystem paths, model selection, and device configuration environment-driven.

## Capability Graph
- @skill/legacy-modernization
- @skill/rag-quality
- @skill/python-release-engineering

## Development Guidelines
- Runtime/Environment: Python with `uv`, reproducible lockfile, explicit dependency groups.
- Architecture Patterns: Domain/application/ports/infrastructure, adapter-based interfaces, compatibility wrappers during migration.
- Safety: Add characterization tests before invasive extractions.

## Delegation & Boundaries

### Nested Rules

| Rule Scope | Location |
| :--- | :--- |
| `src/bhodi_platform/` | [src/bhodi_platform/AGENTS.md](/src/bhodi_platform/AGENTS.md) |
