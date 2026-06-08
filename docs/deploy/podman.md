# Deploy Bhodi with Podman

This guide covers building and running the Bhodi API container with **Podman** and `podman-compose`.

## Prerequisites

- Podman installed — see <https://podman.io/docs/installation>
- `podman-compose` installed (`pip install podman-compose` or your distro's package)
- An OpenAI API key (only required if you use the `openai` embedding or LLM providers; switch to `mock` / `in_memory` providers to run fully offline)

## Build and run

### Option 1: Compose stack (API + standalone Chroma)

The included `podman-compose.yml` starts two services: a `bhodi-api` container built from `Containerfile`, and a `chromadb/chroma:latest` vector store.

```bash
export OPENAI_API_KEY="sk-..."
podman-compose up --build
```

You can also put the key in a `.env` file at the project root:

```text
OPENAI_API_KEY=sk-...
```

```bash
podman-compose --env-file .env up -d
```

### Option 2: Build and run the API container directly

```bash
podman build -f Containerfile -t bhodi .
podman run --rm -p 8000:8000 -e OPENAI_API_KEY="sk-..." bhodi
```

## Verify the stack

| Service | Endpoint | Notes |
|---------|----------|-------|
| Bhodi API | <http://localhost:8000> | Main application (Swagger UI at `/docs`) |
| Bhodi health | <http://localhost:8000/health> | Used by the container's `HEALTHCHECK` |
| Chroma DB | <http://localhost:8080> | Standalone Chroma service, internal port `8000` |

Quick smoke test:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"source": "./document.pdf"}'
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}'
```

## Logs

```bash
# All services
podman-compose logs -f

# Only the API
podman-compose logs -f bhodi-api
```

## Stop the stack

```bash
podman-compose down
```

To also drop the Chroma named volume (wipes all indexed vectors):

```bash
podman-compose down -v
```

## Environment variables

The compose file and `Containerfile` use the following variables:

| Variable | Where it is read | Default | Description |
|----------|------------------|---------|-------------|
| `OPENAI_API_KEY` | `podman-compose.yml` → `bhodi-api` | — | Required by the OpenAI adapters; not needed for `mock` providers |
| `BHODI_API_HOST` | `podman-compose.yml` → `bhodi-api` | `0.0.0.0` in the container | Binds the API to all interfaces inside the container |
| `BHODI_API_PORT` | `podman-compose.yml` → `bhodi-api` | `8000` | Port the API listens on inside the container |
| `BHODI_API_SOURCE_ROOT` | API app (`interfaces/api/app.py`) | unset | When set, constrains local file ingest for `POST /documents` to that root directory |

> The `bhodi-api` image is built without `BHODI_API_HOST` / `BHODI_API_PORT` baked in; the compose file is what sets them. If you run the image directly with `podman run`, the API defaults from `server.py` apply (`127.0.0.1:8000`).

Provider selection (`BHODI_PARSER_PROVIDER`, `BHODI_CHUNKER_PROVIDER`, `BHODI_EMBEDDING_PROVIDER`, `BHODI_VECTOR_STORE_PROVIDER`, `BHODI_LLM_PROVIDER`, `BHODI_CONVERSATION_PROVIDER`) is supported via standard `BhodiConfig` env handling; the compose file does not pin any of them, so defaults from `application/config.py` apply (`pypdf`, `recursive`, `openai`, `chroma`, `openai`, `volatile`).

## Data persistence

- The compose file mounts `./data` on the host to `/app/data` inside the `bhodi-api` container. That path is the default location for ChromaDB's on-disk persistence when you select the `chroma` vector store and an on-disk `persist_directory`; it also keeps any uploaded documents across container restarts.
- A named Podman volume `chroma-data` (declared under `volumes:`) is used for the standalone `chroma` service. The compose file maps it to `/chroma/chroma` inside the `chroma` container and sets `IS_PERSISTENT=TRUE` plus `PERSIST_DIRECTORY=/chroma/chroma` for that service.

## Image split: pinned Python client vs. latest server

The `bhodi-api` image installs **`chromadb==0.5.23`** from `pyproject.toml` — the Python client is intentionally pinned because 1.0.0–1.5.9 are affected by CVE-2026-45829 (a critical pre-auth code injection; no upstream fix as of 2026-06-05).

The standalone `chroma` service in `podman-compose.yml` is a **different image**: it pulls `chromadb/chroma:latest` from Docker Hub and runs the server in its own container on port `8080`. As a consequence, the API process and the vector store can end up on different Chroma versions. If you care about a single known-good version everywhere, either:

- run an embedded Chroma by selecting the Python `chroma` provider with an on-disk `persist_directory` and removing the `chroma` service from the compose file, or
- pin a specific `chromadb/chroma:<tag>` image in `podman-compose.yml` and track upstream issue #6717 for a 1.5.10+ fix.

Always put the standalone Chroma service behind a reverse proxy with authentication, regardless of which version it is on.

## Troubleshooting

### `Permission denied` on `./data`

Podman containers run rootless by default. Make sure the host user owns the directory:

```bash
mkdir -p ./data
podman unshare chown 1000:1000 ./data
```

### Port already in use

If port `8000` or `8080` is taken, edit the `ports:` mappings in `podman-compose.yml`:

```yaml
ports:
  - "8001:8000"   # bhodi-api
  - "8081:8000"   # chroma
```

### `OPENAI_API_KEY not set` inside the container

The compose file expects the variable to be present in the shell or in `.env` before you run `podman-compose up`. Either `export` it or create `.env` with `OPENAI_API_KEY=sk-...`. If you do not want to use OpenAI, set `BHODI_EMBEDDING_PROVIDER=mock` and `BHODI_LLM_PROVIDER=mock` (and optionally `BHODI_VECTOR_STORE_PROVIDER=in_memory`) in the same `.env` file.

### `/health` returns 503

The container's health check, the API process, and the `bhodi` adapters may not all be ready yet. Re-run `curl http://localhost:8000/health` after a few seconds; if it stays 503, inspect the API logs (`podman-compose logs -f bhodi-api`) for the underlying adapter error reported in the `services` map.

### Container cannot reach the Chroma service

Confirm both containers are on the same network and that `bhodi-api` depends on `chroma`. From inside the API container you should be able to resolve `chroma` and connect to port `8000` (the internal port) — `podman-compose.yml` already wires this up.

## Files reference

| File | Purpose |
|------|---------|
| `Containerfile` | `bhodi-api` image (Python 3.11-slim, uv-based install, non-root user) |
| `podman-compose.yml` | Multi-service orchestration: `bhodi-api` (built locally) + `chromadb/chroma:latest` |
