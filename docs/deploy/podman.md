# Deploy Bhodi with Podman

This guide covers building and running the Bhodi API container with **Podman** and `podman-compose`.

## Prerequisites

- Podman installed — see <https://podman.io/docs/installation>
- `podman-compose` installed (`pip install podman-compose` or your distro's package)
- An OpenAI API key (only required if you use the `openai` embedding or LLM providers; switch to `mock` / `in_memory` providers to run fully offline)

## Build and run

### Option 1: Compose stack (API only — ChromaDB runs in embedded mode)

The included `podman-compose.yml` runs a single `bhodi-api` container built from `Containerfile`. ChromaDB runs in embedded mode inside the API process.

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

- The compose file mounts `./data` on the host to `/app/data` inside the `bhodi-api` container. That path is the default location for ChromaDB's on-disk persistence when you select the `chroma` vector store and an on-disk `persist_directory`; it also keeps any uploaded documents across container restarts. No other named volumes are declared.

## Troubleshooting

### `Permission denied` on `./data`

Podman containers run rootless by default. Make sure the host user owns the directory:

```bash
mkdir -p ./data
podman unshare chown 1000:1000 ./data
```

### Port already in use

If port `8000` is taken, edit the `ports:` mapping in `podman-compose.yml`:

```yaml
ports:
  - "8001:8000"   # bhodi-api
```

### `OPENAI_API_KEY not set` inside the container

The compose file expects the variable to be present in the shell or in `.env` before you run `podman-compose up`. Either `export` it or create `.env` with `OPENAI_API_KEY=sk-...`. If you do not want to use OpenAI, set `BHODI_EMBEDDING_PROVIDER=mock` and `BHODI_LLM_PROVIDER=mock` (and optionally `BHODI_VECTOR_STORE_PROVIDER=in_memory`) in the same `.env` file.

### `/health` returns 503

The container's health check, the API process, and the `bhodi` adapters may not all be ready yet. Re-run `curl http://localhost:8000/health` after a few seconds; if it stays 503, inspect the API logs (`podman-compose logs -f bhodi-api`) for the underlying adapter error reported in the `services` map.

## Files reference

| File | Purpose |
|------|---------|
| `Containerfile` | `bhodi-api` image (Python 3.11-slim-bookworm, uv-based install, non-root user) |
| `podman-compose.yml` | Single-service orchestration: `bhodi-api` (built locally) with embedded ChromaDB |
