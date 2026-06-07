#!/usr/bin/env python3
"""Quality ratchet for first-party static analysis debt."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final


EXIT_SUCCESS: Final[int] = 0
EXIT_REGRESSION: Final[int] = 1
EXIT_TOOL_ERROR: Final[int] = 2

METRIC_ORDER: Final[tuple[str, ...]] = (
    "ruff_check_total",
    "ruff_format_files",
    "mypy_errors_total",
    "deptry_issues_total",
)

RUFF_FORMAT_RE = re.compile(r"(?P<count>\d+) files? would be reformatted")
MYPY_RE = re.compile(r"Found (?P<count>\d+) errors? in (?P<files>\d+) files?")
DEPTRY_RE = re.compile(r"DEP\d{3}")


class RatchetError(RuntimeError):
    """Raised when a tool invocation or parse fails unexpectedly."""


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined_output(self) -> str:
        parts = [part.strip() for part in (self.stdout, self.stderr) if part.strip()]
        return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Path to the versioned baseline JSON file.",
    )
    parser.add_argument(
        "--write-baseline",
        type=Path,
        help="Write the current metrics to the given baseline JSON path.",
    )
    args = parser.parse_args()

    if args.baseline is None and args.write_baseline is None:
        parser.error("at least one of --baseline or --write-baseline is required")

    return args


def fail(message: str) -> int:
    summary = f"## Quality ratchet\n\n- ❌ {message}\n"
    print(summary, file=sys.stderr)
    write_step_summary(summary)
    return EXIT_TOOL_ERROR


def write_step_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    with Path(summary_path).open("a", encoding="utf-8") as handle:
        handle.write(markdown)
        if not markdown.endswith("\n"):
            handle.write("\n")


def get_repo_root() -> Path:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RatchetError("`git` is required to locate the repository root.") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() or error.stdout.strip() or str(error)
        raise RatchetError(f"Unable to determine repository root: {stderr}") from error

    repo_root = output.stdout.strip()
    if not repo_root:
        raise RatchetError("`git rev-parse --show-toplevel` returned an empty path.")
    return Path(repo_root)


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def run_command(
    argv: list[str],
    *,
    repo_root: Path,
    allowed_returncodes: set[int],
) -> CommandResult:
    try:
        completed = subprocess.run(
            argv,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RatchetError(f"Required tool is missing: {argv[0]}") from error
    except OSError as error:
        raise RatchetError(f"Failed to execute {' '.join(argv)}: {error}") from error

    result = CommandResult(
        argv=tuple(argv),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if result.returncode not in allowed_returncodes:
        details = result.combined_output or "<no output>"
        raise RatchetError(
            f"Command failed unexpectedly with exit {result.returncode}: {' '.join(argv)}\n{details}"
        )
    return result


def parse_ruff_check(result: CommandResult) -> int:
    payload = result.stdout.strip()
    if not payload:
        raise RatchetError("ruff check did not produce JSON output.")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RatchetError(f"Unable to parse ruff check JSON output: {error}") from error
    if not isinstance(parsed, list):
        raise RatchetError("ruff check JSON output must be a list.")
    return len(parsed)


def parse_ruff_format(result: CommandResult) -> int:
    output = result.combined_output
    match = RUFF_FORMAT_RE.search(output)
    if match:
        return int(match.group("count"))
    if "already formatted" in output or (result.returncode == 0 and not output):
        return 0
    raise RatchetError(f"Unable to parse ruff format output.\n{output or '<no output>'}")


def parse_mypy(result: CommandResult) -> int:
    output = result.combined_output
    match = MYPY_RE.search(output)
    if match:
        return int(match.group("count"))
    if "Success: no issues found" in output or (result.returncode == 0 and not output):
        return 0
    raise RatchetError(f"Unable to parse mypy output.\n{output or '<no output>'}")


def parse_deptry(result: CommandResult) -> int:
    output = result.combined_output
    count = len(DEPTRY_RE.findall(output))
    if count > 0 or result.returncode == 0:
        return count
    raise RatchetError(f"Unable to parse deptry output.\n{output or '<no output>'}")


def collect_metrics(repo_root: Path) -> dict[str, int]:
    metrics: dict[str, int] = {}

    ruff_check = run_command(
        ["ruff", "check", "src/", "tests/", "--output-format", "json"],
        repo_root=repo_root,
        allowed_returncodes={0, 1},
    )
    metrics["ruff_check_total"] = parse_ruff_check(ruff_check)

    ruff_format = run_command(
        ["ruff", "format", "--check", "src/", "tests/"],
        repo_root=repo_root,
        allowed_returncodes={0, 1},
    )
    metrics["ruff_format_files"] = parse_ruff_format(ruff_format)

    mypy = run_command(
        ["mypy", "src/", "--show-error-codes", "--hide-error-context", "--no-pretty"],
        repo_root=repo_root,
        allowed_returncodes={0, 1},
    )
    metrics["mypy_errors_total"] = parse_mypy(mypy)

    deptry = run_command(
        ["deptry", "src/"],
        repo_root=repo_root,
        allowed_returncodes={0, 1},
    )
    metrics["deptry_issues_total"] = parse_deptry(deptry)

    return metrics


def load_baseline(path: Path) -> dict[str, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RatchetError(f"Baseline file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise RatchetError(f"Baseline file is not valid JSON: {path}: {error}") from error

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise RatchetError(f"Baseline file is missing a metrics object: {path}")

    baseline: dict[str, int] = {}
    for metric_name in METRIC_ORDER:
        value = metrics.get(metric_name)
        if not isinstance(value, int):
            raise RatchetError(
                f"Baseline metric `{metric_name}` must be an integer in {path}."
            )
        baseline[metric_name] = value

    return baseline


def get_tool_versions(repo_root: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name, argv in {
        "ruff": ["ruff", "--version"],
        "mypy": ["mypy", "--version"],
        "deptry": ["deptry", "--version"],
    }.items():
        result = run_command(argv, repo_root=repo_root, allowed_returncodes={0})
        versions[name] = (result.stdout or result.stderr).strip()
    return versions


def write_baseline(path: Path, repo_root: Path, metrics: dict[str, int]) -> None:
    sha_result = run_command(["git", "rev-parse", "HEAD"], repo_root=repo_root, allowed_returncodes={0})
    payload = {
        "schema_version": 1,
        "generated_from_sha": sha_result.stdout.strip(),
        "generated_on": date.today().isoformat(),
        "tool_versions": get_tool_versions(repo_root),
        "metrics": {metric_name: metrics[metric_name] for metric_name in METRIC_ORDER},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_summary(baseline: dict[str, int], current: dict[str, int]) -> tuple[str, bool]:
    has_regression = False
    lines = [
        "## Quality ratchet",
        "",
        "| Metric | Baseline | Current | Delta | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for metric_name in METRIC_ORDER:
        baseline_value = baseline[metric_name]
        current_value = current[metric_name]
        delta = current_value - baseline_value
        is_regression = current_value > baseline_value
        has_regression = has_regression or is_regression
        status = "❌ regressed" if is_regression else "✅"
        lines.append(
            "| {metric} | {baseline} | {current} | {delta:+d} | {status} |".format(
                metric=metric_name,
                baseline=baseline_value,
                current=current_value,
                delta=delta,
                status=status,
            )
        )

    overall = "FAIL" if has_regression else "PASS"
    lines.extend(["", f"Overall ratchet status: **{overall}**"])
    return "\n".join(lines) + "\n", has_regression


def main() -> int:
    args = parse_args()

    try:
        repo_root = get_repo_root()
        current_metrics = collect_metrics(repo_root)

        if args.write_baseline is not None:
            output_path = resolve_path(args.write_baseline, repo_root)
            write_baseline(output_path, repo_root, current_metrics)
            print(f"Wrote baseline to {output_path}")

        if args.baseline is None:
            return EXIT_SUCCESS

        baseline_path = resolve_path(args.baseline, repo_root)
        baseline_metrics = load_baseline(baseline_path)
        summary, has_regression = build_summary(baseline_metrics, current_metrics)
        print(summary, end="")
        write_step_summary(summary)
        return EXIT_REGRESSION if has_regression else EXIT_SUCCESS
    except RatchetError as error:
        return fail(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
