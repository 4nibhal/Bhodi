# Bhodi

**Production-ready RAG framework with clean hexagonal architecture.**

Bhodi is a Python library for building retrieval-augmented generation (RAG) applications. It provides a clean separation between domain logic, infrastructure adapters, and transport interfaces.

## Features

- **Clean Architecture**: Hexagonal architecture with domain, application, ports, infrastructure, and interfaces layers
- **Pluggable Adapters**: Swap embeddings, vector stores, LLMs, chunkers, parsers, and conversation memory
- **Multiple Interfaces**: REST API (FastAPI) on port 8000, CLI, or integrate as a library
- **Type-Safe**: Full type hints with Pydantic models
- **Async-First**: All operations are async
- **Observable**: OpenTelemetry spans built into adapters
- **Health Check**: Real health check verifying embedding, vector_store, and llm connectivity
- **Rate Limiting**: 100 requests/minute per IP on REST API
- **Podman Deploy**: Ready-to-use Containerfile and podman-compose.yml

## Installation

```bash
pip install bhodi
```

Or install with extras:

```bash
pip install bhodi[local-llm]  # Ollama support
pip install bhodi[tui]        # Textual TUI
pip install bhodi[telemetry]   # OpenTelemetry
pip install bhodi[all]         # All extras
```

### Development

```bash
git clone https://github.com/your-org/bhodi
cd bhodi
uv sync
uv run pytest  # Run tests
```

## Entry Points

- `bhodi` — CLI tool (`index`, `query`, `health`)
- `bhodi-api` — REST API server (FastAPI, port 8000)
- `bhodi-index` — Indexing worker / batch indexer

## Quick Start

### CLI

```bash
# Index a document
bhodi index ./document.pdf

# Query the index
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

async def main():
    # Configure adapters
    config = BhodiConfig(
        embedding={"provider": "openai", "model": "text-embedding-3-small"},
        vector_store={"provider": "chroma", "persist_directory": "./data/chroma"},
        chunker={"provider": "recursive", "chunk_size": 512},
        llm={"provider": "openai", "model": "gpt-4o-mini"},
    )

    # Build application
    container = Container.from_config(config)
    app = container.build()

    # Index documents
    document_id = await app.index_document({
        "source": "./document.pdf",
        "metadata": {"author": "Test"}
    })

    # Query
    response = await app.query({
        "question": "What is this about?",
        "top_k": 5
    })
    print(response.answer_text)

asyncio.run(main())
```

### REST API

```bash
# Start server
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

## Deploy with Podman

Bhodi includes a `Containerfile` and `podman-compose.yml` for containerized deployment:

```bash
# Build and run with Podman Compose
podman-compose up --build

# Or manually with Podman
podman build -f Containerfile -t bhodi .
podman run -p 8000:8000 --env-file .env bhodi
```

## Rate Limiting

The REST API enforces **100 requests per minute per IP**. Exceeding this limit returns HTTP 429 (Too Many Requests). Rate limits apply to all endpoints except `/health`.

## Architecture

```
bhodi_platform/
├── domain/           # Pure business logic
│   ├── entities.py   # Document, Chunk, Query, Answer
│   ├── value_objects.py  # DocumentId, ChunkId, Citation
│   └── exceptions.py # Domain exceptions
├── application/     # Use cases & orchestration
│   ├── config.py     # Configuration schema
│   ├── facade.py     # BhodiApplication
│   └── models.py     # Request/Response models
├── ports/           # Abstract interfaces (Protocols)
│   ├── embedding.py  # EmbeddingPort
│   ├── vector_store.py  # VectorStorePort
│   ├── chunker.py   # ChunkerPort
│   ├── parser.py    # ParserPort
│   ├── llm.py       # LLMPort
│   └── conversation_memory.py  # ConversationMemoryPort
├── infrastructure/  # Concrete adapters
│   ├── container.py # Dependency injection (Container.from_config + build)
│   ├── telemetry.py # OpenTelemetry spans
│   ├── embedding/   # OpenAI (text-embedding-3-small), Mock
│   ├── vector_store/ # Chroma (persistent), InMemory (ephemeral)
│   ├── chunker/     # FixedSize, Recursive (character splitting)
│   ├── parser/      # PyPDF, Mock
│   ├── llm/         # OpenAI (gpt-4o-mini), Ollama (llama3.2), Mock
│   └── conversation_memory/  # Volatile (in-memory), Mock
└── interfaces/     # Transport adapters
    ├── api/         # FastAPI server (port 8000)
    └── cli/         # CLI commands (bhodi)
```

## Configuration

All configuration is done via `BhodiConfig`:

```python
from bhodi_platform.application.config import (
    BhodiConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    LLMConfig,
    ChunkerConfig,
    ConversationConfig,
    ParserConfig,
)

config = BhodiConfig(
    embedding=EmbeddingConfig(
        provider="openai",  # or "mock"
        model="text-embedding-3-small",
        dimensions=1536,
    ),
    vector_store=VectorStoreConfig(
        provider="chroma",  # or "in_memory"
        persist_directory="./data/chroma",
        collection_name="bhodi",
    ),
    chunker=ChunkerConfig(
        provider="recursive",  # or "fixed_size"
        chunk_size=512,
        overlap=64,
    ),
    llm=LLMConfig(
        provider="openai",  # or "ollama", "mock"
        model="gpt-4o-mini",
        temperature=0.7,
    ),
    parser=ParserConfig(
        provider="pypdf",  # or "mock"
    ),
    conversation=ConversationConfig(
        provider="volatile",  # or "mock"
        max_history=50,
    ),
)
```

### Environment Variables

```bash
# OpenAI (required for OpenAI adapters)
export OPENAI_API_KEY="sk-..."

# Ollama (for local LLM)
export OLLAMA_BASE_URL="http://localhost:11434"
```

## Available Adapters

### Embeddings

| Provider | Model | Notes |
|----------|-------|-------|
| `openai` | text-embedding-3-small | Requires OPENAI_API_KEY |
| `mock` | - | For testing |

### Vector Stores

| Provider | Notes |
|----------|-------|
| `chroma` | Persistent storage (on-disk) |
| `in_memory` | Ephemeral, for testing |

### LLMs

| Provider | Model | Notes |
|----------|-------|-------|
| `openai` | gpt-4o-mini | Requires OPENAI_API_KEY |
| `ollama` | llama3.2 | Local, requires Ollama server |
| `mock` | - | For testing |

### Chunkers

| Provider | Notes |
|----------|-------|
| `fixed_size` | Fixed character/byte chunks |
| `recursive` | Recursive character splitting |

### Parsers

| Provider | Notes |
|----------|-------|
| `pypdf` | Extracts text from PDF files |
| `mock` | For testing |

### Conversation Memory

| Provider | Notes |
|----------|-------|
| `volatile` | In-memory, per-process |
| `mock` | For testing |

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (verifies embedding, vector_store, llm) |
| POST | `/documents` | Index a document |
| DELETE | `/documents/{id}` | Delete a document |
| POST | `/query` | Query the index |

### Request/Response Models

```python
# IndexDocumentRequest
{
    "source": "path/to/document.pdf",
    "metadata": {"key": "value"},
    "chunk_size": 512,
    "overlap": 64
}

# IndexDocumentResponse
{
    "document_id": "uuid",
    "chunk_count": 42
}

# QueryRequest
{
    "question": "What is this about?",
    "conversation_id": "optional-session-id",
    "top_k": 5,
    "temperature": 0.7
}

# QueryResponse
{
    "answer_text": "The document is about...",
    "citations": [
        {
            "chunk_id": "doc-uuid:0",
            "text": "First part of source...",
            "source_document": "document.pdf",
            "page": 1
        }
    ],
    "conversation_id": "session-id"
}
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=bhodi_platform --cov-report=html

# Run specific test suites
uv run pytest tests/unit/
uv run pytest tests/contract/
uv run pytest tests/e2e/
uv run pytest tests/integration/
```

## Roadmap

Features planned but **not yet implemented**:

- Persistent conversation memory (currently only in-memory volatile)
- Authentication / API key management
- Semantic chunker
- Anthropic LLM adapter

## License

Apache 2.0 - see LICENSE file for details.

<!-- AIWF-GENERATED:START -->
- AIWF generated summary:
  - repo_intent: `consumer`
  - setup_mode: `steady_state`
  - governance_mode: `delegated`
  - interaction_mode: `standard`
  - runtime_policy: `track_compiled`
  - platforms: `opencode`
<!-- AIWF-GENERATED:END -->
