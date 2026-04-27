from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bhodi API server (FastAPI + Uvicorn)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("BHODI_API_HOST", "127.0.0.1"),
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("BHODI_API_PORT", "8000")),
        help="Port to bind (default: 8000)",
    )
    args = parser.parse_args()

    uvicorn.run(
        "bhodi_platform.interfaces.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
    )
