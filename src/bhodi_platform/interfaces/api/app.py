"""
FastAPI application factory.

Creates and configures the Bhodi API server.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from bhodi_platform.application.config import BhodiConfig
from bhodi_platform.application.facade import BhodiApplication
from bhodi_platform.infrastructure.container import Container


# Module-level state
_state: dict = {"app": None, "bhodi_app": None}

# Rate limiting state
_rate_limit_requests: dict[str, list[float]] = {}
_rate_limit_lock = asyncio.Lock()
RATE_LIMIT_MAX = 100
RATE_LIMIT_WINDOW = 60  # seconds


async def _rate_limit_middleware(request: Request, call_next):
    """Simple in-memory rate limiter. Skip /health endpoint."""
    if request.url.path == "/health":
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    async with _rate_limit_lock:
        timestamps = _rate_limit_requests.get(client_ip, [])
        # Clean old timestamps
        timestamps = [ts for ts in timestamps if now - ts < RATE_LIMIT_WINDOW]
        if len(timestamps) >= RATE_LIMIT_MAX:
            _rate_limit_requests[client_ip] = timestamps
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
        timestamps.append(now)
        _rate_limit_requests[client_ip] = timestamps

    return await call_next(request)


def create_app(config: BhodiConfig | None = None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        config: Optional BhodiConfig. If not provided, uses defaults.

    Returns:
        Configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup/shutdown."""
        cfg = config or BhodiConfig()
        container = Container(cfg)
        _state["bhodi_app"] = container.build()
        yield
        _state["bhodi_app"] = None

    app = FastAPI(
        title="Bhodi API",
        description="Production-ready RAG framework with clean hexagonal architecture",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Register middleware
    app.middleware("http")(_rate_limit_middleware)

    # Include routers
    from bhodi_platform.interfaces.api.routes import health, indexing, query

    app.include_router(health.router, tags=["health"])
    app.include_router(indexing.router, tags=["documents"])
    app.include_router(query.router, tags=["query"])

    # Global exception handler for domain errors
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions."""
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc)
                if not isinstance(exc, Exception)
                else "Internal server error"
            },
        )

    _state["app"] = app
    return app


def get_bhodi_app() -> BhodiApplication:
    """Get the BhodiApplication instance."""
    app = _state.get("bhodi_app")
    if app is None:
        raise RuntimeError("Application not initialized")
    return app
