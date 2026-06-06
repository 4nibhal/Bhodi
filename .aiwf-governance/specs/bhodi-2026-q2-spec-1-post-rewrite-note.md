# Spec #1 (bhodi-2026-q2-security-and-major-dependency-bumps) — Post-Rewrite Reconciliation Note

**Date**: 2026-06-05
**Triggered by**: Rewrite of commit history to apply `git-excellence` nomenclature to the 5 authored commits (4 PR head commits + 1 consolidation merge)

## What happened

The 4 PRs (#25, #26, #27, #28) were opened, approved, and merged into main with proper git-excellence bodies but the original commit headers used a non-allowed type (`deps`) and contained spec-internal numbering (e.g., "(PR #1 of Spec #1)") in the header. The history was rewritten using `git filter-branch` to:

1. Replace the `deps(...)` type with the conventional `build(deps)` type (per git-excellence allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert)
2. Remove the spec-internal numbering from the headers
3. Improve the bodies to be more declarative (motivation over "what")

## Impact on the 4 PRs

The 4 PRs remain in GitHub's MERGED state, but their `merge_commit_sha` fields point to the original (pre-rewrite) auto-merge commits, which no longer exist in the rewritten history. This means the "View merge commit" link on each PR's page returns 404.

**This reconciliation PR provides a fresh, valid `merge_commit_sha` for the 4 PRs to reference.** The 4 PR bodies will be updated with a "Reconciliation" note pointing to this PR.

## Current commit map (in rewritten main)

| Old SHA | New SHA | Description |
|---------|---------|-------------|
| 9d44ba5 | e859d03 | PR #25 — build(deps): bump aiohttp.../pypdf... |
| 0cc49b2 | 288da1f | PR #26 — build(deps): bump core stack... |
| 8032a3b | 9850f46 | PR #27 — build(deps): bump test/lint stack... |
| c2312cb | 249f76c | PR #28 — build(deps): bump optional and build deps... |
| cf1bc6e | 56e6e29 | Consolidation merge |

The old SHAs remain reachable via the reflog for ~90 days; rollback is possible via `git reset --hard cf1bc6e@{'1'}` if needed (but would revert the rewrite entirely).

## Future-proofing

Future Spec #1 follow-ups (Spec #2 — CI hardening max) will use proper `git-excellence` headers from the start. The pre-rewrite commit messages are documented in `.aiwf-governance/specs/bhodi-2026-q2-spec-1-original-commit-messages.md` for audit purposes.
