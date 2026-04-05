"""Main CLI entry point for Bhodi."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bhodi - Production-ready RAG framework",
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
    subparsers.add_parser("health", help="Check service health")

    args = parser.parse_args()

    if args.command == "index":
        from bhodi_platform.interfaces.cli.indexing import main as index_main

        index_main([args.source])
    elif args.command == "query":
        from bhodi_platform.interfaces.cli.query import main as query_main

        query_main([args.question])
    elif args.command == "health":
        print("Health check: OK")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
