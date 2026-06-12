#!/usr/bin/env python3
"""
Sync tracked sub-agent definitions to the runtime mirror.

The tracked source-of-truth for sub-agent definitions lives in
`opencode-flows/agent/*.md`. The OpenCode runtime consumes copies at
`.opencode/agents/` (gitignored, machine-local). This script keeps the
runtime mirror in sync with the tracked source after a fresh clone,
after pulling, or after any change to the tracked definitions.

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
import shutil
import sys
from pathlib import Path


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
            print(f"  [=] unchanged: {src.name}")
            unchanged += 1
        else:
            if dry_run:
                print(f"  [>] would copy: {src.name}")
            else:
                shutil.copy2(src, dst)
                print(f"  [+] synced:    {src.name}")
            synced += 1

    # 2) Remove stale files that no longer exist in the source.
    for dst in sorted(dst_dir.glob("*.md")):
        if not (src_dir / dst.name).exists():
            if dry_run:
                print(f"  [x] would remove: {dst.name}")
            else:
                dst.unlink()
                print(f"  [-] removed:    {dst.name} (no longer in source)")
            removed += 1

    return {"synced": synced, "unchanged": unchanged, "removed": removed}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync opencode-flows/agent/*.md to .opencode/agents/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without modifying files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the runtime mirror is out of sync with source.",
    )
    args = parser.parse_args()

    if args.dry_run and args.check:
        print("Error: --dry-run and --check are mutually exclusive.", file=sys.stderr)
        return 2

    try:
        repo_root = find_repo_root(Path(__file__).parent)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    src_dir = repo_root / "opencode-flows" / "agent"
    dst_dir = repo_root / ".opencode" / "agents"

    print(f"Source: {src_dir.relative_to(repo_root)}")
    print(f"Dest:   {dst_dir.relative_to(repo_root)}")
    print(f"Mode:   {'DRY-RUN' if args.dry_run else 'CHECK' if args.check else 'APPLY'}")
    print()

    try:
        counters = sync_agents(src_dir, dst_dir, dry_run=args.dry_run or args.check)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Are you running this from inside the bodhi-rag repo?", file=sys.stderr)
        return 2

    total_changed = counters["synced"] + counters["removed"]
    print()
    print(
        f"Summary: synced={counters['synced']} "
        f"unchanged={counters['unchanged']} removed={counters['removed']}"
    )

    if args.check:
        if total_changed > 0:
            print("Drift detected: runtime mirror is out of sync with source.", file=sys.stderr)
            return 1
        print("In sync.")
        return 0

    if args.dry_run and total_changed > 0:
        print("(dry-run; no files modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
