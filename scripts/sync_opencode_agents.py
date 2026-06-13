#!/usr/bin/env python3
"""
Sync tracked sub-agent definitions to the runtime mirror.

The tracked source-of-truth for sub-agent definitions lives in
`opencode-flows/agent/*.md`. The OpenCode runtime consumes copies at
`.opencode/agents/` (gitignored, machine-local). This script keeps the
runtime mirror in sync with the tracked source after a fresh clone,
after pulling, or after any change to the tracked definitions.

Additionally, the script patches the tracked `.opencode/opencode.json`
so the `agent.system-architect.permission.task` allowlist covers every
agent in source. Newly-added agents get `"allow"` (the default
permissive stance for tracked agents; per-machine tightening lives
in `.opencode/opencode.local.json`).

Idempotent: running multiple times produces the same state.
Deterministic: no network calls, no timestamps, no random ordering.
Safe: --dry-run prints planned actions without modifying files.

Usage:
    python scripts/sync_opencode_agents.py [--dry-run] [--check]

Exit codes:
    0  Sync complete (or already in sync). With --check, also means
       no drift was detected.
    1  Drift detected (--check mode only).
    2  Tool error (source dir missing, not in repo, etc.).
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

# The path inside opencode.json where delegated-agent allowlist lives.
# Centralized so the contract is documented in one place.
_TASK_PERMISSION_PATH = ("agent", "system-architect", "permission", "task")


def find_repo_root(start: Path) -> Path:
    """Walk up from `start` until a `.git` directory is found."""
    cur = start.resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / ".git").is_dir():
            return candidate
    msg = "Could not locate repo root (no .git directory found in any parent)"
    raise FileNotFoundError(msg)


def sync_agents(
    src_dir: Path,
    dst_dir: Path,
    *,
    dry_run: bool,
) -> dict[str, int]:
    """
    Mirror `src_dir/*.md` into `dst_dir/`, removing stale files.

    Returns a dict with counters: synced, unchanged, removed.
    """
    if not src_dir.is_dir():
        msg = f"Source directory not found: {src_dir}"
        raise FileNotFoundError(msg)

    dst_dir.mkdir(parents=True, exist_ok=True)

    synced = unchanged = removed = 0

    # 1) Copy or overwrite each source file into the destination.
    for src in sorted(src_dir.glob("*.md")):
        dst = dst_dir / src.name
        if dst.is_file() and filecmp.cmp(src, dst, shallow=False):
            unchanged += 1
        else:
            if dry_run:
                pass
            else:
                shutil.copy2(src, dst)
            synced += 1

    # 2) Remove stale files that no longer exist in the source.
    for dst in sorted(dst_dir.glob("*.md")):
        if not (src_dir / dst.name).exists():
            if dry_run:
                pass
            else:
                dst.unlink()
            removed += 1

    return {"synced": synced, "unchanged": unchanged, "removed": removed}


def _agent_names_in_source(src_dir: Path) -> set[str]:
    """Return the set of agent names derived from the tracked source files."""
    return {p.stem for p in src_dir.glob("*.md")}


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        result: dict = json.load(fh)
        return result


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")


def patch_opencode_task_permission(  # noqa: PLR0912
    src_dir: Path,
    config_path: Path,
    *,
    dry_run: bool,
) -> dict[str, int]:  # one return per outcome; collapsing would hurt readability
    """
    Patch the opencode.json task-permission allowlist to cover every source agent.

    Every agent file in `src_dir` gets a `"allow"` entry in
    `agent.system-architect.permission.task` of `config_path`; stale
    entries (agent removed from source) are dropped. Returns counters:
    added, already_present, removed.
    """
    if not config_path.is_file():
        msg = f"opencode.json not found: {config_path}"
        raise FileNotFoundError(msg)

    names = _agent_names_in_source(src_dir)
    config = _read_json(config_path)

    # Walk to the task-permission map, creating intermediates if absent.
    cursor: dict = config
    for key in _TASK_PERMISSION_PATH[:-1]:
        cursor = cursor.setdefault(key, {})
    task_perm = cursor.setdefault(_TASK_PERMISSION_PATH[-1], {})
    # Make sure the wildcard-deny default is in place.
    task_perm.setdefault("*", "deny")

    added = already_present = removed = 0
    # Add new entries.
    for name in sorted(names):
        if name not in task_perm:
            if dry_run:
                pass
            else:
                task_perm[name] = "allow"
            added += 1
        else:
            already_present += 1
    # Remove stale entries (agent removed from source).
    for name in sorted(set(task_perm) - names - {"*"}):
        if dry_run:
            pass
        else:
            del task_perm[name]
        removed += 1

    if added or removed:
        if dry_run:
            pass
        else:
            _write_json(config_path, config)

    return {"added": added, "already_present": already_present, "removed": removed}


def main() -> int:  # noqa: PLR0911  # 4 returns per CLI outcome
    parser = argparse.ArgumentParser(
        description=(
            "Sync opencode-flows/agent/*.md to .opencode/agents/ "
            "and patch .opencode/opencode.json task.permission accordingly."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without modifying files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the runtime mirror or opencode.json is out of sync with source.",
    )
    args = parser.parse_args()

    if args.dry_run and args.check:
        return 2

    try:
        repo_root = find_repo_root(Path(__file__).parent)
    except FileNotFoundError:
        return 2

    src_dir = repo_root / "opencode-flows" / "agent"
    dst_dir = repo_root / ".opencode" / "agents"
    config_path = repo_root / ".opencode" / "opencode.json"


    try:
        counters = sync_agents(src_dir, dst_dir, dry_run=args.dry_run or args.check)
    except FileNotFoundError:
        return 2

    try:
        perm_counters = patch_opencode_task_permission(
            src_dir,
            config_path,
            dry_run=args.dry_run or args.check,
        )
    except FileNotFoundError:
        return 2

    total_changed = (
        counters["synced"]
        + counters["removed"]
        + perm_counters["added"]
        + perm_counters["removed"]
    )

    if args.check:
        if total_changed > 0:
            return 1
        return 0

    if args.dry_run and total_changed > 0:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
