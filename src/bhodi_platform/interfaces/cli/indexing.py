"""CLI command for indexing documents."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TextIO

from bhodi_platform.application.config import BhodiConfig
from bhodi_platform.application.facade import BhodiApplication
from bhodi_platform.application.models import IndexDocumentRequest
from bhodi_platform.infrastructure.container import Container


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for index command."""
    parser = argparse.ArgumentParser(
        description="Index documents for later querying.",
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path to document or directory to index.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Target chunk size (default from config).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Overlap between chunks (default from config).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Additional metadata as JSON string.",
    )
    return parser


async def run_index(
    source: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    metadata: dict | None = None,
) -> str:
    """Run index operation and return result string."""
    config = BhodiConfig()
    container = Container(config)
    app = container.build()

    request = IndexDocumentRequest(
        source=source,
        metadata=metadata or {},
        chunk_size=chunk_size,
        overlap=overlap,
    )

    try:
        response = await app.index_document(request)
        return (
            f"Indexed {response.chunk_count} chunks from {source}. "
            f"Document ID: {response.document_id}"
        )
    except Exception as e:
        return f"Error indexing {source}: {e}"


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    output = stdout or sys.stdout

    # Parse metadata if provided
    metadata = None
    if args.metadata:
        import json

        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in --metadata: {args.metadata}", file=sys.stderr)
            sys.exit(1)

    result = asyncio.run(
        run_index(
            source=args.source,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            metadata=metadata,
        )
    )
    print(result, file=output)


if __name__ == "__main__":
    main()
