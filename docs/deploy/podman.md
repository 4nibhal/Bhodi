# Deploy Bhodi with Podman

This guide covers building and running Bhodi containers using **Podman** and `podman-compose`.

## Prerequisites

- Podman installed ([instructions](https://podman.io/docs/installation))
- `podman-compose` installed (`pip install podman-compose` or your distro package)
- An OpenAI API key

## Quick start

### 1. Build the image

```bash
podman build -t bhodi -f Containerfile .
```

### 2. Start the stack

Copy the example environment file and add your key:

```bash
export OPENAI_API_KEY="sk-..."
podman-compose up -d
```

Or create a `.env` file in the project root:

```text
OPENAI_API_KEY=sk-...
```

Then run:

```bash
podman-compose --env-file .env up -d
```

### 3. Verify services

| Service | Endpoint | Notes |
|---------|----------|-------|
| Bhodi API | http://localhost:8000 | Main application |
| Bhodi Health | http://localhost:8000/health | Container health check |
| Chroma DB | http://localhost:8080 | Vector database (internal port 8000) |

### 4. View logs

```bash
# All services
podman-compose logs -f

# Only the API
podman-compose logs -f bhodi-api
```

### 5. Stop the stack

```bash
podman-compose down
```

To also remove the Chroma named volume:

```bash
podman-compose down -v
```

## Required environment variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and generation | `sk-...` |
| `BHODI_API_HOST` | Host the API binds to (use `0.0.0.0` inside containers) | `0.0.0.0` |
| `BHODI_API_PORT` | Port the API listens on | `8000` |

## Optional / project-specific variables

Bhodi also reads these variables at runtime if you need to override defaults:

| Variable | Description |
|----------|-------------|
| `BHODI_INDEX_PERSIST_DIRECTORY` | Path for index persistence |
| `BHODI_CONVERSATION_PERSIST_DIRECTORY` | Path for conversation persistence |
| `BHODI_EMBEDDINGS_MODEL` | Embedding model name |
| `BHODI_LOCAL_MODEL` | Local LLM model path |

## Data persistence

- `./data/` on the host is mounted to `/app/data` inside the `bhodi-api` container.  
  This keeps ChromaDB local files and any uploads across container restarts.
- A named Podman volume `chroma-data` is used for the standalone `chroma` service.

## Troubleshooting

### Permission denied on `./data`
Podman containers run rootless by default. Ensure your user owns the `./data` directory:

```bash
mkdir -p ./data
podman unshare chown 1000:1000 ./data
```

### Port already in use
If port `8000` or `8080` is taken, edit the port mappings in `podman-compose.yml`:

```yaml
ports:
  - "8001:8000"   # bhodi-api
  - "8081:8000"   # chroma
```

## Files reference

| File | Purpose |
|------|---------|
| `Containerfile` | Bhodi API container image definition |
| `podman-compose.yml` | Multi-service orchestration (API + Chroma) |
