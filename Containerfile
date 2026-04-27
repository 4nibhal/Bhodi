# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

# Install build dependencies for native packages (numpy, pypdf, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Astral's Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests and source code
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install production dependencies and the project into a virtual environment.
# --frozen ensures uv.lock is not modified.
# --no-dev skips development dependencies.
RUN uv sync --frozen --no-dev

# ------------------------------------------------------------------------------
# Runtime stage
# ------------------------------------------------------------------------------
FROM python:3.11-slim

# Create a non-privileged user (UID 1000)
RUN useradd --create-home --uid 1000 --shell /bin/bash bhodi

WORKDIR /app

# Copy the virtual environment and source code from the builder stage.
# uv sync installs the project in editable mode, so /app/src must remain available.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Make venv binaries available on PATH (includes bhodi-api, bhodi, bhodi-index)
ENV PATH="/app/.venv/bin:$PATH"

# Create data directory for ChromaDB local persistence and assign ownership
RUN mkdir -p /app/data && chown -R bhodi:bhodi /app

USER bhodi

# Expose the API port
EXPOSE 8000

# Health check using only the Python standard library (no curl needed)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Entry point defined in pyproject.toml as "bhodi-api"
ENTRYPOINT ["bhodi-api"]
