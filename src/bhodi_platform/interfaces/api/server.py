from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "bhodi_platform.interfaces.api.app:create_app",
        factory=True,
        host=os.getenv("BHODI_API_HOST", "127.0.0.1"),
        port=int(os.getenv("BHODI_API_PORT", "8000")),
    )
