---
scope: "opencode-flows/"
type: "rules"
role: "OpenCode Flow Source-of-Truth"
parent: "/"
priority: medium
metadata:
  system: "aiwf"
  mirrors_to: ".opencode/agents/"
  description: >-
    Tracked source-of-truth for OpenCode sub-agent definitions. Files in
    `opencode-flows/agent/*.md` are mirrored to `.opencode/agents/*.md` for
    runtime use. The mirror is a copy (not a symlink); the tracked source
    is the canonical contract.
---

# Rules: OpenCode Flow Source-of-Truth

## Context & Responsibility
This scope owns the tracked, source-of-truth definitions of OpenCode sub-agents. The runtime copies in `.opencode/agents/` are gitignored and machine-local; the files under `opencode-flows/agent/` are the contract.

## Operational Standards
- One `.md` file per sub-agent, named after the agent (e.g., `backend-architect.md`).
- Frontmatter must include `description:`, `mode: subagent`, and an explicit `tools:` allow-list.
- Body sections follow the template in `.opencode/skills/opencode-agent-creator/assets/AGENT-TEMPLATE.md` (Critical Boundaries, Operational Workflow).
- When a sub-agent gains or loses a tool, update both this source file and the corresponding `opencode.json` `task.permission` entry in the same change.

## Capability Graph
- @skill/opencode-agent-creator

## Delegation & Boundaries

### Nested Rules

| Rule Scope | Location |
|------------|----------|
| `opencode-flows/agent/` | inline per-agent markdown headers (no separate AGENTS.md yet) |
