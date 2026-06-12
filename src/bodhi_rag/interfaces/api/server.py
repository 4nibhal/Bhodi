from __future__ import annotations

import argparse
import os

import uvicorn

from bodhi_rag.application.config import ConfigError
from bodhi_rag.application.config_loader import load_bodhi_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bodhi RAG API server (FastAPI + Uvicorn)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("BODHI_API_HOST", "127.0.0.1"),
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("BODHI_API_PORT", "8000")),
        help="Port to bind (default: 8000)",
    )
    args = parser.parse_args()

    # Validate the TOML config (if configured) at startup so the server
    # fails fast with a clear error rather than at the first request.
    try:
        load_bodhi_config()
    except ConfigError as exc:
        msg = f"Config error: {exc}"
        raise SystemExit(msg) from exc

    uvicorn.run(
        "bodhi_rag.interfaces.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
    )
