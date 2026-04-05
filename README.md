# Bhodi

**Production-ready RAG framework with clean hexagonal architecture.**

Bhodi is a Python library for building retrieval-augmented generation (RAG) applications. It provides a clean separation between domain logic, infrastructure adapters, and transport interfaces.

## Features

- **Clean Architecture**: Hexagonal architecture with domain, application, ports, and infrastructure layers
- **Pluggable Adapters**: Swap embeddings (OpenAI, local), vector stores (Chroma), LLMs (OpenAI, Ollama)
- **Multiple Interfaces**: REST API (FastAPI), CLI, or integrate as a library
- **Type-Safe**: Full type hints with Pydantic models
- **Async-First**: All operations are async
- **Observable**: OpenTelemetry instrumentation built-in

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
from bhodi_platform.application.config import BhodiConfig
from bhodi_platform.application.facade import BhodiApplication
from bhodi_platform.infrastructure.container import Container

# Configure adapters
config = BhodiConfig(
    embedding={"provider": "openai", "model": "text-embedding-3-small"},
    vector_store={"provider": "chroma", "persist_directory": "./data/chroma"},
    chunker={"provider": "recursive", "chunk_size": 512},
    llm={"provider": "openai", "model": "gpt-4o-mini"},
)

# Build application
container = Container(config)
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
│   ├── llm.py       # LLMPort
│   └── conversation_memory.py  # ConversationMemoryPort
├── infrastructure/  # Concrete adapters
│   ├── container.py # Dependency injection
│   ├── telemetry.py # OpenTelemetry
│   ├── embedding/   # OpenAI, Local, Mock
│   ├── vector_store/ # Chroma, InMemory
│   ├── chunker/     # FixedSize, Recursive
│   ├── llm/         # OpenAI, Ollama, Mock
│   └── conversation_memory/  # Volatile, Persistent
└── interfaces/     # Transport adapters
    ├── api/         # FastAPI server
    └── cli/         # CLI commands
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
)

config = BhodiConfig(
    embedding=EmbeddingConfig(
        provider="openai",  # or "mock", "local"
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
    conversation=ConversationConfig(
        provider="volatile",  # or "persistent"
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
| `openai` | text-embedding-3-small, text-embedding-3-large | Requires OPENAI_API_KEY |
| `local` | sentence-transformers/* | Uses local models |
| `mock` | - | For testing |

### Vector Stores

| Provider | Notes |
|----------|-------|
| `chroma` | Persistent storage |
| `in_memory` | Ephemeral, for testing |

### LLMs

| Provider | Model | Notes |
|----------|-------|-------|
| `openai` | gpt-4o, gpt-4o-mini | Requires OPENAI_API_KEY |
| `ollama` | llama3.2, mistral, etc. | Local, requires Ollama server |
| `mock` | - | For testing |

### Chunkers

| Provider | Notes |
|----------|-------|
| `fixed_size` | Fixed character/byte chunks |
| `recursive` | Recursive character splitting |

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/documents` | Index a document |
| DELETE | `/documents/{id}` | Delete a document |
| POST | `/query` | Query the index |
| GET | `/conversations/{id}` | Get conversation history |

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
