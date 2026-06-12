"""Main CLI entry point for bodhi-rag."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

from bodhi_rag.application.config_loader import load_bodhi_config


def _load_config_or_exit(config_path: "str | Path | None") -> "object | None":
    """Load the config via `load_bodhi_config` and exit on `ConfigError`.

    Returns the loaded `BhodiConfig` (or None if `config_path` is None and
    the env / default path is not configured). On error, prints a clear
    message to stderr and exits with code 2.
    """
    if config_path is None and "BODHI_CONFIG_PATH" not in os.environ:
        # No config layer requested — let the subcommand build its own
        # default `BhodiConfig()` and the caller's Container do the wiring.
        return None
    try:
        return load_bodhi_config(config_path=config_path)
    except Exception as exc:  # noqa: BLE001 - top-level CLI error path
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(2)


def _health_command() -> int:
    """Probe the live bodhi-rag-api /health endpoint and propagate its state.

    Exit codes:
        0 - the API is healthy (HTTP 200, status=healthy)
        2 - the API is degraded (HTTP 503, status=degraded, or
            HTTP 200 with status != "healthy")
        1 - the API is unreachable, refused the connection, or
            returned an unexpected status

    The CLI is a client of the API, not a parallel process that
    re-instantiates adapters. It uses BODHI_API_HOST and
    BODHI_API_PORT (the same env vars the API server reads) to
    locate the API process.
    """
    host = os.getenv("BODHI_API_HOST", "127.0.0.1")
    port = int(os.getenv("BODHI_API_PORT", "8000"))
    url = f"http://{host}:{port}/health"

    try:
        resp = httpx.get(url, timeout=2.0)
    except httpx.RequestError as exc:
        print(f"Health check: UNREACHABLE ({url}): {exc}", file=sys.stderr)
        return 1

    try:
        data = resp.json()
    except ValueError:
        print(
            f"Health check: INVALID RESPONSE ({url}): "
            f"status={resp.status_code} body={resp.text!r}",
            file=sys.stderr,
        )
        return 1

    status_value = data.get("status", "unknown")
    version_value = data.get("version", "unknown")
    services = data.get("services", {})

    if resp.status_code == 200 and status_value == "healthy":
        print(
            f"Health check: OK ({url})  "
            f"version={version_value}  services={services}"
        )
        return 0

    if status_value == "degraded" or resp.status_code == 503:
        print(
            f"Health check: DEGRADED ({url})  "
            f"status={resp.status_code}  body={data}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Health check: UNEXPECTED ({url})  "
        f"status={resp.status_code}  body={data}",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="bodhi-rag - Production-ready RAG framework",
    )
    # Top-level config flag. The flag is parsed before subcommand dispatch
    # so the loaded config is available to every subcommand via `load_bodhi_config`.
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help=(
            "Path to a bodhi.toml config file. Overrides BODHI_CONFIG_PATH "
            "and ./bodhi.toml. See docs/configuration.md for the schema."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Index subcommand
    index_parser = subparsers.add_parser(
        "index",
        help="Index documents for querying",
    )
    index_parser.add_argument("source", type=str, help="Path to document or directory")

    # Query subcommand
    query_parser = subparsers.add_parser(
        "query",
        help="Query indexed documents",
    )
    query_parser.add_argument("question", type=str, help="Question to ask")

    # Health subcommand
    subparsers.add_parser("health", help="Probe the live API /health endpoint")

    args = parser.parse_args()

    # Load the TOML config (if any) up front; subcommands that need a
    # `BhodiConfig` will use it, others (like `health`) ignore it.
    _load_config_or_exit(args.config)

    if args.command == "index":
        from bodhi_rag.interfaces.cli.indexing import main as index_main

        index_main([args.source])
        return 0
    if args.command == "query":
        from bodhi_rag.interfaces.cli.query import main as query_main

        query_main([args.question])
        return 0
    if args.command == "health":
        return _health_command()

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
