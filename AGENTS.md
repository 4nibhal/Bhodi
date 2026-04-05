---
scope: "/"
type: "rules"
role: "Bhodi Root Governance"
priority: critical
metadata:
  system: "aiwf"
  product_name: "Bhodi"
  product_posture: "backend-first document processing platform"
  legacy_surfaces: ["src/bhodi_doc_analyzer", "src/indexer"]
  non_product_paths: ["ai-workflow", ".opencode", ".aiwf"]
---

# Rules: Bhodi Root

## Context & Responsibility
This file is the root governance contract for Bhodi. The product is a backend-first document processing and retrieval platform. Terminal UX, local developer runtimes, and AI workflow infrastructure are supporting concerns, not the product core.

## Product Boundary
- Product code lives under `src/` and future product-facing docs/tests.
- `ai-workflow/`, `.opencode/`, and `.aiwf/` are developer operating model infrastructure and must not become runtime dependencies of the shipped product.
- Legacy packages in `src/bhodi_doc_analyzer/` and `src/indexer/` are transitional surfaces to be extracted into the target backend architecture.

## Architectural Direction
- Prefer backend-first service boundaries over TUI-centric orchestration.
- New business logic belongs in `src/bhodi_platform/` using domain/application/ports/infrastructure separation.
- Treat UI, CLI, API, workers, and TUI as adapters over application services.
- Keep configuration explicit, typed, and environment-driven.
- No import-time side effects for model loading, vector store creation, filesystem writes, or network initialization.

## Operational Standards
- Environment: Use `uv` as the canonical Python workflow for local development and CI.
- Verification: Favor `uv sync`, targeted tests, packaging checks, and CI parity over ad hoc local-only flows.
- Quality Gates: Refactors must add characterization coverage before changing legacy behavior materially.
- Release Safety: Product packaging, CI, Dependabot, and observability are part of the architecture, not optional polish.
- Separation of Concerns: Product work and AIWF/dev-infra work should stay logically separated unless the change explicitly couples both.

## Capability Graph
- @skill/legacy-modernization
- @skill/rag-quality
- @skill/python-release-engineering
- @skill/github-automation
- @skill/rules-creator
- @skill/skill-creator
- @skill/opencode-agent-creator

### Auto-invoke Skills

When performing these actions, ALWAYS invoke the corresponding skill FIRST:

| Action | Skill |
|--------|-------|
| After creating/modifying a skill | [`skill-sync`](/.opencode/skills/skill-sync/SKILL.md) |
| After creating/modifying a skill | [`skill-sync`](/skills/skill-sync/SKILL.md) |
| Before creating a commit | [`git-excellence`](/.opencode/skills/git-excellence/SKILL.md) |
| Before creating a commit or PR | [`git-excellence`](/.opencode/skills/git-excellence/SKILL.md) |
| Changing chunking, retrieval, reranking, prompting, or answer grounding | [`rag-quality`](/skills/rag-quality/SKILL.md) |
| Creating PRs, release workflows, Dependabot config, or GitHub automation policies | [`github-automation`](/skills/github-automation/SKILL.md) |
| Creating new OpenCode sub-agents | [`opencode-agent-creator`](/.opencode/skills/opencode-agent-creator/SKILL.md) |
| Creating new OpenCode sub-agents | [`opencode-agent-creator`](/skills/opencode-agent-creator/SKILL.md) |
| Creating or scaffolding new skills | [`skill-creator`](/.opencode/skills/skill-creator/SKILL.md) |
| Creating or scaffolding new skills | [`skill-creator`](/skills/skill-creator/SKILL.md) |
| Defining new rules or scaffolding AGENTS.md | [`rules-creator`](/.opencode/skills/rules-creator/SKILL.md) |
| Defining new rules or scaffolding AGENTS.md | [`rules-creator`](/skills/rules-creator/SKILL.md) |
| During Pull Request creation | [`git-excellence`](/.opencode/skills/git-excellence/SKILL.md) |
| Modifying AGENTS.md structure or adding new rules | [`rules-sync`](/.opencode/skills/rules-sync/SKILL.md) |
| Modifying AGENTS.md structure or adding new rules | [`rules-sync`](/skills/rules-sync/SKILL.md) |
| Preparing builds, packaging, CI, lockfile policy, or release automation | [`python-release-engineering`](/skills/python-release-engineering/SKILL.md) |
| Refactoring legacy Bhodi modules into the target backend architecture | [`legacy-modernization`](/skills/legacy-modernization/SKILL.md) |
| Regenerate AGENTS.md Auto-invoke tables (sync.sh) | [`skill-sync`](/.opencode/skills/skill-sync/SKILL.md) |
| Regenerate AGENTS.md Auto-invoke tables (sync.sh) | [`skill-sync`](/skills/skill-sync/SKILL.md) |
| Running AIWF init or refining bootstrap intake behavior | [`bootstrap`](/.opencode/skills/bootstrap/SKILL.md) |
| Starting a new feature or change that requires design thinking | [`brainstorm`](/.opencode/skills/brainstorm/SKILL.md) |
| Starting a spec-driven development workflow | [`spec-workflow`](/.opencode/skills/spec-workflow/SKILL.md) |
| Synchronize all systems (OpenCode runtime, AGENTS.md, skill/rule metadata) | [`system-sync`](/.opencode/skills/system-sync/SKILL.md) |
| Troubleshoot why a skill is missing from AGENTS.md auto-invoke | [`skill-sync`](/.opencode/skills/skill-sync/SKILL.md) |
| Troubleshoot why a skill is missing from AGENTS.md auto-invoke | [`skill-sync`](/skills/skill-sync/SKILL.md) |
| User describes intent to build something without a clear spec | [`brainstorm`](/.opencode/skills/brainstorm/SKILL.md) |
| When an agent needs to operate the AIWF CLI or machine API | [`aiwf-cli-operator`](/.opencode/skills/aiwf-cli-operator/SKILL.md) |
| When detecting staged changes | [`git-excellence`](/.opencode/skills/git-excellence/SKILL.md) |

## Specialized Sub-agents (OpenCode Flows)

Source definitions live in `opencode-flows/agent/` and synchronize into `.opencode/agents/` for runtime use.

| Domain / Trigger | Agent |
|------------------|-------|
| Backend architecture, module boundaries, migration design | [`backend-architect`](opencode-flows/agent/backend-architect.md) |
| RAG retrieval, chunking, reranking, citations, evals | [`rag-systems-engineer`](opencode-flows/agent/rag-systems-engineer.md) |
| Python quality, packaging, testing, runtime safety | [`python-quality-engineer`](opencode-flows/agent/python-quality-engineer.md) |
| GitHub workflows, Dependabot, PR/release operations | [`github-ops-engineer`](opencode-flows/agent/github-ops-engineer.md) |

## Platform Native Agents

| Agent | Purpose |
|-------|---------|
| `explore` | Fast repository analysis and codebase discovery. |
| `general` | General multi-step execution support. |
| `web-researcher` | External documentation and current ecosystem research. |

## Delegation & Boundaries

### Nested Rules

| Rule Scope | Location |
| :--- | :--- |
| `/skills/` | [skills/AGENTS.md](/skills/AGENTS.md) |
| `opencode-flows` | [opencode-flows/AGENTS.md](/opencode-flows/AGENTS.md) |
| `src/` | [src/AGENTS.md](/src/AGENTS.md) |
| `tests/` | [tests/AGENTS.md](/tests/AGENTS.md) |

## Nested Rules

| Rule Scope | Location |
|------------|----------|
| `src/` | [`src/AGENTS.md`](src/AGENTS.md) |
| `src/bhodi_platform/` | [`src/bhodi_platform/AGENTS.md`](src/bhodi_platform/AGENTS.md) |
| `src/bhodi_platform/interfaces/tui/` | [`src/bhodi_platform/interfaces/tui/AGENTS.md`](src/bhodi_platform/interfaces/tui/AGENTS.md) |
| `tests/` | [`tests/AGENTS.md`](tests/AGENTS.md) |
| `skills/` | [`skills/AGENTS.md`](skills/AGENTS.md) |
| `opencode-flows/` | [`opencode-flows/AGENTS.md`](opencode-flows/AGENTS.md) |
