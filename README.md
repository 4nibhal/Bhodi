# Bhodi

[![CI](https://github.com/4nibhal/Bhodi/actions/workflows/ci.yml/badge.svg)](https://github.com/4nibhal/Bhodi/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A backend RAG engine for Python developers.**

Bhodi indexes documents and answers questions using retrieval-augmented generation (RAG). It is built as a modular, hexagonal backend that you can run as a library, a CLI, or a REST API.

> **Status:** Beta. Core RAG pipeline is solid. Auth, persistent memory, and managed SaaS features are not implemented yet. See [Roadmap](#roadmap).

---

## Try it in 30 seconds (no API keys)

```bash
uv tool install bhodi

export BHODI_EMBEDDING_PROVIDER=mock
export BHODI_LLM_PROVIDER=mock
export BHODI_VECTOR_STORE_PROVIDER=in_memory

bhodi index ./document.pdf
bhodi query "What is this document about?"
```

Uses mock adapters. No network calls. No OpenAI account required. Good for exploring the CLI and local testing.

> Or with **pipx**: `pipx install bhodi`

---

## Installation

```bash
uv tool install bhodi
```

Or with **pipx**:

```bash
pipx install bhodi
```

With extras:

```bash
uv tool install bhodi --with bhodi[local-llm]  # Ollama support
uv tool install bhodi --with bhodi[tui]        # Textual TUI
uv tool install bhodi --with bhodi[telemetry]  # OpenTelemetry
uv tool install bhodi --with bhodi[all]        # All extras
```

Or with pipx:

```bash
pipx install bhodi
pipx inject bhodi bhodi[local-llm]
```

> **Why uv/pipx instead of pip?**
> `uv tool install` and `pipx install` install Bhodi in an isolated environment, avoiding dependency conflicts with your system Python or other projects. This is the recommended way to install CLI tools.

### Development

```bash
git clone https://github.com/4nibhal/bhodi.git
cd bhodi
uv sync
uv run pytest
```

---

## Quick Start

### CLI with OpenAI

```bash
export OPENAI_API_KEY="sk-..."

# Index a document
bhodi index ./document.pdf

# Query
bhodi query "What is this document about?"

# Health check
bhodi health
```

### Python API

```python
import asyncio
from bhodi_platform.application.config import BhodiConfig
from bhodi_platform.application.facade import BhodiApplication
from bhodi_platform.infrastructure.container import Container
from bhodi_platform.application.models import IndexDocumentRequest, QueryRequest

async def main():
    config = BhodiConfig(
        embedding={"provider": "openai", "model": "text-embedding-3-small"},
        vector_store={"provider": "chroma", "persist_directory": "./data/chroma"},
        chunker={"provider": "recursive", "chunk_size": 512},
        llm={"provider": "openai", "model": "gpt-4o-mini"},
    )

    container = Container.from_config(config)
    app = container.build()

    response = await app.index_document(
        IndexDocumentRequest(source="./document.pdf", metadata={"author": "Test"})
    )
    print(f"Indexed {response.chunk_count} chunks")

    answer = await app.query(
        QueryRequest(question="What is this about?", top_k=5)
    )
    print(answer.answer_text)

asyncio.run(main())
```

### REST API

```bash
# Start server
export OPENAI_API_KEY="sk-..."
bhodi-api

# In another terminal:
curl http://localhost:8000/health

curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"source": "./document.pdf"}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}'
```

The API enforces **100 requests/minute per IP** (HTTP 429 when exceeded). `/health` is excluded.

---

## Deploy with Podman

```bash
# Requires OPENAI_API_KEY in environment or .env
podman-compose up --build
```

Or manually:

```bash
podman build -f Containerfile -t bhodi .
podman run -p 8000:8000 -e OPENAI_API_KEY="sk-..." bhodi
```

> **Warning:** The API has **no authentication**. Only deploy behind a VPN, reverse proxy with auth, or similar. Do not expose directly to the internet.

---

## Configuration

All behavior is driven through `BhodiConfig`:

```python
from bhodi_platform.application.config import (
    BhodiConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    LLMConfig,
    ChunkerConfig,
    ParserConfig,
    ConversationConfig,
)

config = BhodiConfig(
    embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
    vector_store=VectorStoreConfig(
        provider="chroma",
        persist_directory="./data/chroma",
        collection_name="bhodi",
    ),
    chunker=ChunkerConfig(provider="recursive", chunk_size=512, overlap=64),
    llm=LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.7),
    parser=ParserConfig(provider="pypdf"),
    conversation=ConversationConfig(provider="volatile", max_history=50),
)
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."        # Required for OpenAI adapters
export OLLAMA_BASE_URL="http://localhost:11434"  # Optional, for local LLM
```

---

## Available Adapters

| Component | Providers | Notes |
|-----------|-----------|-------|
| **Embeddings** | `openai` (text-embedding-3-small), `mock` | OpenAI requires API key |
| **Vector Store** | `chroma` (persistent), `in_memory` | Chroma stores on disk |
| **LLM** | `openai` (gpt-4o-mini), `ollama` (llama3.2), `mock` | Ollama needs local server |
| **Chunker** | `fixed_size`, `recursive` | Recursive uses character separators |
| **Parser** | `pypdf`, `mock` | PDF text extraction |
| **Conversation Memory** | `volatile`, `mock` | In-memory only; lost on restart |

Swap adapters by changing the `provider` field in config. No code changes needed.

---

## Architecture

Hexagonal separation:

```
bhodi_platform/
├── domain/          # Entities, value objects, exceptions
├── application/     # Config, facade, use cases
├── ports/           # Protocols (EmbeddingPort, LLMPort, etc.)
├── infrastructure/  # Concrete adapters (OpenAI, Chroma, PyPDF)
└── interfaces/      # FastAPI, CLI
```

- **No import-time side effects.** Adapters initialize lazily.
- **Composable.** Use `Container.from_config()` to wire any combination.
- **Testable.** Mock adapters for deterministic tests.

---

## Development

```bash
uv run pytest              # Full suite (184 tests)
uv run pytest tests/evals  # Quality evals
uv run pytest --cov        # Coverage report
uv build                   # Wheel and sdist
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `OPENAI_API_KEY not set` | Missing env var | `export OPENAI_API_KEY="sk-..."` or set providers to `mock` via env vars |
| `Connection refused` on Chroma | Chroma not running | Start Chroma or use `provider="in_memory"` |
| `Rate limit exceeded` | 100 req/min hit | Wait 60 seconds or implement client-side backoff |
| PDF not parsing | Corrupted or scanned PDF | Ensure text-based PDF; scanned images need OCR (not supported) |
| Ollama timeout | Model loading slowly | Increase timeout or pre-pull model: `ollama pull llama3.2` |

---

## Roadmap

Not implemented yet. Planned for future releases:

- **Authentication / API key management** — Required for any internet-facing deployment
- **Persistent conversation memory** — SQLite or PostgreSQL instead of in-memory
- **Semantic chunker** — Chunking based on meaning, not just characters
- **Anthropic LLM adapter** — Claude support
- **Structured logging and metrics** — Operational observability

---

## License

MIT — See [LICENSE](LICENSE).
