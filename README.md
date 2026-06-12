# Bodhi RAG

[![CI](https://github.com/4nibhal/bodhi-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/4nibhal/bodhi-rag/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A backend RAG engine for Python developers.**

bodhi-rag indexes documents and answers questions using retrieval-augmented generation. It is built as a modular, hexagonal backend that you can run as a library, a CLI, or a REST API — with swappable adapters for embeddings, vector stores, LLMs, chunkers, and parsers, plus pluggable conversation memory.

> **Status:** Beta. The core indexing/query pipeline is functional. **Authentication, persistent conversation memory, and a managed SaaS distribution are not implemented yet.** See [Roadmap](#roadmap).

## How it works

A request flows through a clean `interface → application → ports → adapter` chain. The API or CLI hands an `IndexDocumentRequest` / `QueryRequest` to the `BhodiApplication` facade, which delegates to a `DocumentParserPort`, `ChunkerPort`, `EmbeddingPort`, `VectorStorePort`, and `LLMPort` (plus `ConversationMemoryPort` for multi-turn). Each port is a `typing.Protocol`; concrete adapters (OpenAI, ChromaDB, PyPDF, …) live under `infrastructure/` and are wired by a single `Container`. The result: your business logic never imports a vendor SDK, and any adapter can be swapped by changing one Pydantic config field.

---

## Try it in 30 seconds (no API keys)

```bash
uv tool install bodhi-rag

export BODHI_EMBEDDING_PROVIDER=mock
export BODHI_LLM_PROVIDER=mock
export BODHI_VECTOR_STORE_PROVIDER=in_memory

bodhi-rag index ./document.pdf
bodhi-rag query "What is this document about?"
bodhi-rag health
```

Uses mock adapters. No network calls. No OpenAI account required. Good for exploring the CLI and local testing.

> Or with **pipx**: `pipx install bodhi-rag`

---

## Installation

```bash
uv tool install bodhi-rag
```

Or with **pipx**:

```bash
pipx install bodhi-rag
```

### Optional extras

| Extra | Install command | What it adds |
|-------|-----------------|--------------|
| `bodhi-rag[local-llm]` | `uv tool install bodhi-rag --with bodhi-rag[local-llm]` | `llama-cpp-python==0.3.28`, `ollama==0.6.2` |
| `bodhi-rag[telemetry]` | `uv tool install bodhi-rag --with bodhi-rag[telemetry]` | `opentelemetry-api/sdk/exporter-otlp==1.42.1` |
| `bodhi-rag[all]` | `uv tool install bodhi-rag --with bodhi-rag[all]` | All of the above |

With pipx:

```bash
pipx install bodhi-rag
pipx inject bodhi-rag bodhi-rag[local-llm]    # or bodhi-rag[telemetry], bodhi-rag[all]
```

> **Why uv/pipx instead of pip?** `uv tool install` and `pipx install` install bodhi-rag in an isolated environment, avoiding dependency conflicts with your system Python or other projects.

### Development install

```bash
git clone https://github.com/4nibhal/bodhi-rag.git
cd bodhi-rag
uv sync
uv run pytest
```

---

## Quick start

### CLI (3 entry points)

```bash
export OPENAI_API_KEY="sk-..."

# Index a document (uses mock providers if you set BODHI_*_PROVIDER=mock)
bodhi-rag-index ./document.pdf --chunk-size 512 --overlap 64
bodhi-rag-index ./document.pdf --metadata '{"author": "Test"}'

# Query
bodhi-rag query "What is this document about?"

# Health check
bodhi-rag health
```

You can also drive the top-level `bodhi-rag` command (`bodhi-rag index ...`, `bodhi-rag query ...`, `bodhi-rag health`), and the dedicated `bodhi-rag-api` server:

```bash
bodhi-rag-api --host 0.0.0.0 --port 8000
```

### Python API

```python
import asyncio
from bodhi_rag.application.config import BhodiConfig
from bodhi_rag.application.facade import BhodiApplication
from bodhi_rag.infrastructure.container import Container
from bodhi_rag.application.models import IndexDocumentRequest, QueryRequest


async def main() -> None:
    config = BhodiConfig(
        embedding={"provider": "openai", "model": "text-embedding-3-small"},
        vector_store={"provider": "chroma", "persist_directory": "./data/chroma"},
        chunker={"provider": "recursive", "chunk_size": 512, "overlap": 64},
        llm={"provider": "openai", "model": "gpt-4o-mini"},
        parser={"provider": "pypdf"},
        conversation={"provider": "volatile"},
    )

    app = Container(config).build()  # returns BhodiApplication

    indexed = await app.index_document(
        IndexDocumentRequest(source="./document.pdf", metadata={"author": "Test"})
    )
    print(f"Indexed {indexed.chunk_count} chunks from {indexed.document_id}")

    answer = await app.query(QueryRequest(question="What is this about?", top_k=5))
    print(answer.answer_text)
    for cite in answer.citations:
        print(f"  - {cite.source_document} p.{cite.page}: {cite.text[:80]}")


asyncio.run(main())
```

### REST API

```bash
export OPENAI_API_KEY="sk-..."
bodhi-rag-api
```

In another terminal:

```bash
# Health
curl http://localhost:8000/health

# Index a document
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"source": "./document.pdf"}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}'

# Delete a document
curl -X DELETE http://localhost:8000/documents/<document_id>

# Get conversation history
curl http://localhost:8000/conversations/<conversation_id>
```

The API enforces **100 requests / 60 seconds per IP** (HTTP 429 when exceeded). `/health` is excluded. There is **no authentication** — deploy behind a reverse proxy with auth, a VPN, or similar.

Interactive API docs are served at `/docs` (Swagger UI), `/redoc`, and `/openapi.json`.

---

## Architecture

bodhi-rag uses a hexagonal (ports and adapters) layout. Interfaces call into the `BhodiApplication` facade, which orchestrates ports; concrete adapters are wired in by the `Container`.

```mermaid
flowchart TB
    subgraph Interfaces["Interfaces (adapters)"]
        API["FastAPI app<br/>(bodhi-rag-api)"]
        CLI["argparse CLIs<br/>(bodhi-rag, bodhi-rag-index)"]
    end

    subgraph Application["Application layer"]
        Facade["BhodiApplication<br/>(facade.py)"]
        Config["BhodiConfig<br/>(config.py)"]
        Models["Request / response models<br/>(models.py)"]
    end

    subgraph Domain["Domain layer"]
        Entities["Entities<br/>(Document, Chunk, ...)"]
        VO["Value objects<br/>(DocumentId, ChunkId, ...)"]
        Policy["Policies"]
    end

    subgraph Ports["Ports (Protocols)"]
        EP["EmbeddingPort"]
        VP["VectorStorePort"]
        CP["ChunkerPort"]
        DP["DocumentParserPort"]
        LP["LLMPort"]
        CMPort["ConversationMemoryPort"]
    end

    subgraph Infra["Infrastructure (adapters)"]
        Embed["OpenAI / Mock"]
        Store["Chroma / InMemory"]
        Chunk["FixedSize / Recursive"]
        Parse["PyPDF / Mock"]
        LLM["OpenAI / Ollama / Mock"]
        CMem["Volatile"]
        Container["Container<br/>(composition root)"]
    end

    subgraph Cross["Cross-cutting packages"]
        Indexing["indexing/"]
        Evaluation["evaluation/"]
    end

    API --> Facade
    CLI --> Facade

    Facade --> EP
    Facade --> VP
    Facade --> CP
    Facade --> DP
    Facade --> LP
    Facade --> CMPort

    Container -- wires --> Embed
    Container -- wires --> Store
    Container -- wires --> Chunk
    Container -- wires --> Parse
    Container -- wires --> LLM
    Container -- wires --> CMem

    Embed -. implements .-> EP
    Store -. implements .-> VP
    Chunk -. implements .-> CP
    Parse -. implements .-> DP
    LLM -. implements .-> LP
    CMem -. implements .-> CMPort

    Facade --> Domain
    Facade --> Indexing
    Facade --> Evaluation
```

For the full directory tree and design decisions, see [docs/architecture/overview.md](docs/architecture/overview.md).

---

## Configuration

All runtime behavior is driven through `BhodiConfig` (Pydantic models in `bodhi_rag.application.config`):

```python
from bodhi_rag.application.config import (
    BhodiConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    LLMConfig,
    ChunkerConfig,
    DocumentParserConfig,
    ConversationConfig,
)

config = BhodiConfig(
    embedding=EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        batch_size=100,
    ),
    vector_store=VectorStoreConfig(
        provider="chroma",
        persist_directory="./data/chroma",
        collection_name="bodhi-rag",
    ),
    chunker=ChunkerConfig(provider="recursive", chunk_size=512, overlap=64),
    llm=LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.7),
    parser=DocumentParserConfig(provider="pypdf"),
    conversation=ConversationConfig(provider="volatile", max_history=50),
)
```

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BODHI_API_HOST` | `127.0.0.1` | API server bind host (overridden by `bodhi-rag-api --host`) |
| `BODHI_API_PORT` | `8000` | API server bind port (overridden by `bodhi-rag-api --port`) |
| `BODHI_API_SOURCE_ROOT` | unset | When set, constrains local file ingest for `POST /documents` |
| `BODHI_CONFIG_PATH` | unset | Path to a TOML config file (precedence: CLI > env > TOML > defaults) |
| `OPENAI_API_KEY` | — | Required when any `openai` adapter is selected |
| `BODHI_PARSER_PROVIDER` | `pypdf` | Override parser provider |
| `BODHI_CHUNKER_PROVIDER` | `recursive` | Override chunker provider |
| `BODHI_EMBEDDING_PROVIDER` | `openai` | Override embedding provider |
| `BODHI_VECTOR_STORE_PROVIDER` | `chroma` | Override vector store provider |
| `BODHI_LLM_PROVIDER` | `openai` | Override LLM provider |
| `BODHI_CONVERSATION_PROVIDER` | `volatile` | Override conversation memory provider |

See [`docs/configuration.md`](docs/configuration.md) for the full TOML schema and the cross-encoder worked example.

---

## Available adapters

| Component | Providers | Notes |
|-----------|-----------|-------|
| **Embeddings** | `openai`, `mock` | OpenAI requires `OPENAI_API_KEY`; mock is deterministic and offline |
| **Vector store** | `chroma` (persistent, on-disk), `in_memory` | Chroma uses `chromadb==1.5.9` in embedded mode (see Security note below) |
| **LLM** | `openai`, `ollama`, `mock` | Ollama needs a local server (`ollama serve`) |
| **Chunker** | `fixed_size`, `recursive` | Recursive uses character separators |
| **Document parser** | `pypdf`, `mock` | PDF text extraction via PyPDF |
| **Conversation memory** | `volatile` | In-process only; lost on restart |

Swap adapters by changing the `provider` field. No code changes are required; the `Container` rewires everything.

> **Security note (ChromaDB pinning).** We pin `chromadb==1.5.9`. The server-side CVE-2026-45829 (CVSS 9.3, pre-auth code injection) affects 1.0.0–1.5.9, but bodhi-rag only uses `chromadb.PersistentClient` in embedded mode (`src/bodhi_rag/infrastructure/vector_store/chroma.py`), which never executes the vulnerable code path. We do not deploy the standalone `chromadb/chroma` server. Track upstream issue #6717 for the 1.5.10+ fix.

---

## Security

bodhi-rag treats the dependency surface and the input surface as first-class security concerns. The current posture is built on four layers of defense:

- **Pinned dependencies** — every direct dependency in `pyproject.toml` is locked to an exact `==X.Y.Z` version. Container base images and `podman-compose.yml` tags are pinned too. There are no version ranges. Dependabot opens a PR on every new release; CI verifies the lockfile still resolves. The full audit (dep, pinned version, advisory) lives in [`VERSIONS.md`](VERSIONS.md).
- **`SafeChromaCollection` wrapper** — the ChromaDB adapter is the only vector-store path that touches `chromadb`. It is wrapped in a thin adapter (`src/bodhi_rag/infrastructure/vector_store/chroma.py`) that funnels every call through a small, validated surface so the rest of the system never imports `chromadb` directly. That makes the CVE-2026-45829 vulnerable code path (server-side) unreachable.
- **4-layer defense in depth** — (1) the API rate-limits at 100 req/60s per IP with `/health` excluded; (2) `BODHI_API_SOURCE_ROOT`, when set, confines local-file ingest to that directory and rejects path traversal; (3) input validation lives in Pydantic models (request bodies) and in the FastAPI dependency layer (path params); (4) lazy adapter initialization means the `Container` never opens network connections or allocates GPU on import.
- **No telemetry exfiltration by default** — OpenTelemetry export is opt-in. The default exporter is `console`; nothing leaves the process unless you install `bodhi-rag[telemetry]`, set `exporter="otlp"`, and point at a collector you control.

> The API has **no authentication**. Treat it as a local-or-VPN tool and front it with your own auth proxy if you expose it to a network.

---

## Deploy with Podman

The full guide is in [docs/deploy/podman.md](docs/deploy/podman.md). Quick start:

```bash
export OPENAI_API_KEY="sk-..."
podman-compose up --build
```

Or build and run the API container directly:

```bash
podman build -f Containerfile -t bodhi-rag .
podman run -p 8000:8000 -e OPENAI_API_KEY="sk-..." bodhi-rag
```

The compose stack runs a single `bodhi-rag-api` container built locally from the `Containerfile`. ChromaDB runs in embedded mode inside the same process; no separate vector-store container is used.

> **Warning:** The API has **no authentication**. Only deploy behind a VPN, reverse proxy with auth, or similar. Do not expose directly to the internet.

---

## Documentation

The full docs tree lives under [`docs/`](docs/):

- [Architecture overview](docs/architecture/overview.md) — hexagonal layout, directory tree, data flow, design decisions, telemetry
- [API reference](docs/api/index.md) — every endpoint, request/response shape, status codes, rate-limiting, configuration
- [Deploy with Podman](docs/deploy/podman.md) — `Containerfile`, `podman-compose.yml`, persistence, troubleshooting

For dependency-version history and the CVE audit referenced from the Security section above, see [`VERSIONS.md`](VERSIONS.md).

---

## Development

```bash
uv sync                       # Install locked dev + runtime deps
uv run pytest                 # Full suite (199 tests)
uv run pytest tests/evals     # Quality evaluation suite
uv run pytest --cov           # Coverage report
uv run bandit -r src/         # Bandit security scan
uv run python scripts/quality_ratchet.py --baseline .github/quality-baseline.json
uv build                      # Wheel and sdist
```

CI runs four jobs (`test`, `build`, `security`, `quality`); the `security` job requires both `pip-audit` and `bandit` to pass, and the `quality` job enforces the ratchet baseline.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `OPENAI_API_KEY not set` | Missing env var | `export OPENAI_API_KEY="sk-..."` or set providers to `mock` via `BODHI_*_PROVIDER` env vars |
| ChromaDB fails to start | Persistence path missing or unwritable | Create the directory (`mkdir -p ./data/chroma`) and make sure the API process can read/write it; or set `BODHI_VECTOR_STORE_PROVIDER=in_memory` to run without persistence |
| HTTP `429` from the API | 100 req/60s per IP exceeded | Wait 60 seconds, reduce request rate, or front the API with your own limiter |
| PDF not parsing | Corrupted or scanned PDF | bodhi-rag parses text-based PDFs only; scanned images need OCR, which is not supported |
| Ollama timeout | Model still loading | Pre-pull the model (`ollama pull llama3.2`) and/or increase the client timeout |
| Health check returns 503 | One or more adapters failed to initialize | Check the `services` map in the response body and the server logs for the underlying error |

---

## Roadmap

Not implemented yet; planned for future releases:

- **Authentication / API key management** — required for any internet-facing deployment
- **Persistent conversation memory** — SQLite or PostgreSQL instead of the in-process `volatile` store
- **Semantic chunker** — chunking based on meaning, not just characters
- **Additional LLM adapters** — Anthropic and other providers
- **Operational observability** — structured logging, metrics, and richer OpenTelemetry export
- **Upgrade path to ChromaDB 1.5.10+** — once the upstream CVE-2026-45829 fix ships

---

## AI Workflow

The repo ships with a curated set of OpenCode sub-agents that operate over the codebase. Specialized sub-agents (`backend-architect`, `rag-systems-engineer`, `python-quality-engineer`, `github-ops-engineer`) are tracked in [`opencode-flows/agent/`](opencode-flows/agent/). Platform-native agents (`devops-scripter`, `doc-retriever`, `git-specialist`, `tooling-specialist`, `system-architect`) live in the same tracked directory; the runtime mirror at `.opencode/agents/` is gitignored. Governance rules live in `AGENTS.md` at the repo, `src/`, `src/bodhi_rag/`, `tests/`, and `opencode-flows/` scopes — see the Capability Graph and Auto-invoke Skills tables in each `AGENTS.md` for the contract.

---

## License

MIT — See [LICENSE](LICENSE).
