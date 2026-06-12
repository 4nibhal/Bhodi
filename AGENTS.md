---
scope: "/"
type: "rules"
role: "Bodhi RAG Root Governance"
priority: critical
metadata:
  system: "aiwf"
  product_name: "bodhi-rag"
  product_posture: "backend-first document processing platform"
  non_product_paths: ["ai-workflow", ".opencode", ".aiwf"]
---

# Rules: bodhi-rag Root

## Context & Responsibility
This file is the root governance contract for bodhi-rag. The product is a backend-first document processing and retrieval platform. Terminal UX, local developer runtimes, and AI workflow infrastructure are supporting concerns, not the product core.

## Product Boundary
- Product code lives under `src/` and future product-facing docs/tests.
- `ai-workflow/`, `.opencode/`, and `.aiwf/` are developer operating model infrastructure and must not become runtime dependencies of the shipped product.

## Architectural Direction
- Prefer backend-first service boundaries over TUI-centric orchestration.
- New business logic belongs in `src/bodhi_rag/` using domain/application/ports/infrastructure separation.
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
| Creating new OpenCode sub-agents | [`opencode-agent-creator`](/.opencode/skills/opencode-agent-creator/SKILL.md) |
| Creating new OpenCode sub-agents | [`opencode-agent-creator`](/skills/opencode-agent-creator/SKILL.md) |
| Creating or scaffolding new skills | [`skill-creator`](/.opencode/skills/skill-creator/SKILL.md) |
| Creating or scaffolding new skills | [`skill-creator`](/skills/skill-creator/SKILL.md) |
| Defining new rules or scaffolding AGENTS.md | [`rules-creator`](/.opencode/skills/rules-creator/SKILL.md) |
| Defining new rules or scaffolding AGENTS.md | [`rules-creator`](/skills/rules-creator/SKILL.md) |
| During Pull Request creation | [`git-excellence`](/.opencode/skills/git-excellence/SKILL.md) |
| Modifying AGENTS.md structure or adding new rules | [`rules-sync`](/.opencode/skills/rules-sync/SKILL.md) |
| Modifying AGENTS.md structure or adding new rules | [`rules-sync`](/skills/rules-sync/SKILL.md) |
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

## Platform Native Sub-agents (OpenCode Flows)

Source definitions live in `opencode-flows/agent/` and mirror to `.opencode/agents/` (gitignored, runtime copy). The `system-architect` is a `mode: primary` orchestrator; the rest are `mode: subagent`.

| Agent | Mode | Purpose |
|-------|------|---------|
| `devops-scripter` | subagent | Automation scripts, file bulk operations, log/text parsing. |
| `doc-retriever` | subagent | Verifying technical facts and consulting official documentation. |
| `git-specialist` | subagent | Local Git operations (status, diff, commit, branch). |
| `tooling-specialist` | subagent | Auditing and maintaining infrastructure, CI/CD, repo structure. |
| `system-architect` | primary | High-level orchestrator; subsumes plan/build roles. |

## Delegation & Boundaries

## Nested Rules

| Rule Scope | Location |
|------------|----------|
| `src/` | [`src/AGENTS.md`](src/AGENTS.md) |
| `src/bodhi_rag/` | [`src/bodhi_rag/AGENTS.md`](src/bodhi_rag/AGENTS.md) |
| `tests/` | [`tests/AGENTS.md`](tests/AGENTS.md) |
| `opencode-flows/` | [`opencode-flows/AGENTS.md`](opencode-flows/AGENTS.md) |
