# Deploy bodhi-rag with Podman

This guide covers building and running the bodhi-rag API container with **Podman** and `podman-compose`.

## Prerequisites

- Podman installed — see <https://podman.io/docs/installation>
- `podman-compose` installed (`pip install podman-compose` or your distro's package)
- An OpenAI API key (only required if you use the `openai` embedding or LLM providers; switch to `mock` / `in_memory` providers to run fully offline)

## Build and run

### Option 1: Compose stack (API only — ChromaDB runs in embedded mode)

The included `podman-compose.yml` runs a single `bodhi-rag-api` container built from `Containerfile`. ChromaDB runs in embedded mode inside the API process.

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
podman build -f Containerfile -t bodhi-rag .
podman run --rm -p 8000:8000 -e OPENAI_API_KEY="sk-..." bodhi-rag
```

## Verify the stack

| Service | Endpoint | Notes |
|---------|----------|-------|
| bodhi-rag API | <http://localhost:8000> | Main application (Swagger UI at `/docs`) |
| bodhi-rag health | <http://localhost:8000/health> | Used by the container's `HEALTHCHECK` |

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
podman-compose logs -f bodhi-rag-api
```

## Stop the stack

```bash
podman-compose down
```


## Environment variables

The compose file and `Containerfile` use the following variables:

| Variable | Where it is read | Default | Description |
|----------|------------------|---------|-------------|
| `OPENAI_API_KEY` | `podman-compose.yml` → `bodhi-rag-api` | — | Required by the OpenAI adapters; not needed for `mock` providers |
| `BODHI_API_HOST` | `podman-compose.yml` → `bodhi-rag-api` | `0.0.0.0` in the container | Binds the API to all interfaces inside the container |
| `BODHI_API_PORT` | `podman-compose.yml` → `bodhi-rag-api` | `8000` | Port the API listens on inside the container |
| `BODHI_API_SOURCE_ROOT` | API app (`interfaces/api/app.py`) | unset | When set, constrains local file ingest for `POST /documents` to that root directory |
| `BODHI_CONFIG_PATH` | API app | unset | Path to a TOML config file (`bodhi.toml`) |

> The `bodhi-rag-api` image is built without `BODHI_API_HOST` / `BODHI_API_PORT` baked in; the compose file is what sets them. If you run the image directly with `podman run`, the API defaults from `server.py` apply (`127.0.0.1:8000`).

Provider selection (`BODHI_PARSER_PROVIDER`, `BODHI_CHUNKER_PROVIDER`, `BODHI_EMBEDDING_PROVIDER`, `BODHI_VECTOR_STORE_PROVIDER`, `BODHI_LLM_PROVIDER`, `BODHI_CONVERSATION_PROVIDER`) is supported via standard `BhodiConfig` env handling; the compose file does not pin any of them, so defaults from `application/config.py` apply (`pypdf`, `recursive`, `openai`, `chroma`, `openai`, `volatile`).

## Data persistence

- The compose file mounts `./data` on the host to `/app/data` inside the `bodhi-rag-api` container. That path is the default location for ChromaDB's on-disk persistence when you select the `chroma` vector store and an on-disk `persist_directory`; it also keeps any uploaded documents across container restarts. No other named volumes are declared.

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
  - "8001:8000"   # bodhi-rag-api
```

### `OPENAI_API_KEY not set` inside the container

The compose file expects the variable to be present in the shell or in `.env` before you run `podman-compose up`. Either `export` it or create `.env` with `OPENAI_API_KEY=sk-...`. If you do not want to use OpenAI, set `BODHI_EMBEDDING_PROVIDER=mock` and `BODHI_LLM_PROVIDER=mock` (and optionally `BODHI_VECTOR_STORE_PROVIDER=in_memory`) in the same `.env` file.

### `/health` returns 503

The container's health check, the API process, and the `bodhi-rag` adapters may not all be ready yet. Re-run `curl http://localhost:8000/health` after a few seconds; if it stays 503, inspect the API logs (`podman-compose logs -f bodhi-rag-api`) for the underlying adapter error reported in the `services` map.

## Files reference

| File | Purpose |
|------|---------|
| `Containerfile` | `bodhi-rag-api` image (Python 3.11-slim-bookworm, uv-based install, non-root user) |
| `podman-compose.yml` | Single-service orchestration: `bodhi-rag-api` (built locally) with embedded ChromaDB |
