# Bhodi Architecture Overview

## High-Level Design

Bhodi is a **RAG (Retrieval-Augmented Generation) framework** built with **hexagonal architecture** (also known as ports and adapters pattern).

The core principle is **dependency inversion**: domain logic doesn't depend on infrastructure - instead, infrastructure implements domain-defined interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Interfaces Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │  FastAPI │  │   CLI    │  │  Worker  │                     │
│  │  Server  │  │ Commands │  │  Queue   │                     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                     │
└───────┼──────────────┼──────────────┼────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Application Layer                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   BhodiApplication                        │    │
│  │  - index_document()  - query()  - health_check()        │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│     Ports     │  │     Ports     │  │     Ports     │
│  (Protocols)  │  │  (Protocols)  │  │  (Protocols)  │
│  EmbeddingPort  │  │VectorStorePort│  │  ChunkerPort  │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Infrastructure│  │ Infrastructure│  │ Infrastructure│
│    Adapters   │  │    Adapters   │  │    Adapters   │
│   OpenAI,     │  │   Chroma,      │  │  FixedSize,   │
│   Local       │  │   InMemory    │  │  Recursive    │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Directory Structure

```
bhodi_platform/
├── domain/                    # Pure business logic
│   ├── entities.py           # Document, Chunk, Query, Answer, ConversationTurn
│   ├── value_objects.py      # DocumentId, ChunkId, ConversationId, Citation
│   ├── services.py           # Domain services (optional)
│   └── exceptions.py        # Domain exceptions
│
├── application/             # Use cases and orchestration
│   ├── config.py            # Pydantic configuration models
│   ├── facade.py            # BhodiApplication (main entry point)
│   ├── indexing.py         # IndexDocumentUseCase
│   ├── query.py            # QueryAnswerUseCase
│   └── models.py          # Request/Response DTOs
│
├── ports/                   # Abstract interfaces (Protocols)
│   ├── embedding.py         # EmbeddingPort
│   ├── vector_store.py      # VectorStorePort
│   ├── chunker.py          # ChunkerPort
│   ├── document_parser.py   # DocumentParserPort
│   ├── llm.py              # LLMPort
│   └── conversation_memory.py  # ConversationMemoryPort
│
├── infrastructure/         # Concrete implementations
│   ├── container.py         # Dependency injection
│   ├── telemetry.py         # OpenTelemetry setup
│   ├── tracing.py          # @traced decorator
│   ├── embedding/          # OpenAI, Local, Mock
│   ├── vector_store/       # Chroma, InMemory
│   ├── document_parser/    # PyPDF, Mock
│   ├── chunker/           # FixedSize, Recursive
│   ├── llm/               # OpenAI, Ollama, Mock
│   └── conversation_memory/  # Volatile, Persistent
│
└── interfaces/             # Transport adapters
    ├── api/               # FastAPI server
    │   ├── app.py         # App factory
    │   └── routes/        # Route handlers
    └── cli/               # CLI commands
```

## Core Concepts

### 1. Domain Layer

The domain layer contains **pure business logic** with no external dependencies:

- **Entities**: `Document`, `Chunk`, `Query`, `Answer`, `ConversationTurn`
- **Value Objects**: `DocumentId`, `ChunkId`, `ConversationId`, `Citation`, `EmbeddingVector`
- **Exceptions**: Domain-specific errors

Rules:
- Zero imports from other bhodi_platform layers
- Zero external dependencies (no langchain, chromadb, etc.)
- All validation in `__post_init__` methods

### 2. Ports (Interfaces)

Ports define **what** the domain needs, not **how** it's implemented:

```python
class EmbeddingPort(Protocol):
    """Protocol for embedding generation."""
    
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...
    async def dimensions(self) -> int: ...
```

Using `Protocol` instead of `ABC` allows structural typing - any class with these methods fulfills the contract.

### 3. Adapters (Infrastructure)

Adapters implement ports with concrete technology:

```python
class OpenAIEmbeddingsAdapter:
    """OpenAI implementation of EmbeddingPort."""
    
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
    
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Call OpenAI API
        ...
```

### 4. Dependency Injection

The `Container` wires ports to adapters based on configuration:

```python
config = BhodiConfig(
    embedding={"provider": "openai", "model": "text-embedding-3-small"},
    vector_store={"provider": "chroma"},
)

container = Container(config)
app = container.build()  # Returns BhodiApplication with adapters wired
```

## Data Flow

### Indexing Pipeline

```
User Upload (PDF)
    │
    ▼
DocumentParserPort.parse()
    │
    ▼
Document (text + metadata)
    │
    ▼
ChunkerPort.chunk()
    │
    ▼
List[Chunk]
    │
    ▼
EmbeddingPort.embed_documents()
    │
    ▼
List[Tuple[Chunk, EmbeddingVector]]
    │
    ▼
VectorStorePort.add()
    │
    ▼
Persisted Index
```

### Query Pipeline

```
User Query
    │
    ▼
EmbeddingPort.embed_query()
    │
    ▼
EmbeddingVector
    │
    ▼
VectorStorePort.search()
    │
    ▼
List[RetrievedDocument] (ranked)
    │
    ▼
LLMPort.generate_with_context()
    │
    ▼
Answer (with Citations)
    │
    ▼
Response
```

## Configuration

All configuration is Pydantic-based:

```python
class EmbeddingConfig(BaseModel):
    provider: str
    model: str | None = None
    dimensions: int | None = None
    batch_size: int = 100
    extra: dict[str, Any] = {}
```

Provider-specific config goes in `extra`:

```python
config = EmbeddingConfig(
    provider="ollama",
    model="llama3.2",
    extra={"base_url": "http://localhost:11434"}
)
```

## Key Design Decisions

### 1. Protocol-Based Ports

Using `Protocol` instead of `ABC`:
- **Pros**: Structural typing, no inheritance coupling, easier testing
- **Cons**: IDE autocomplete less precise (but improving)

### 2. Async-First

All port methods are async:
- Non-blocking I/O for better concurrency
- Adapter can wrap sync libraries with `asyncio.to_thread()`

### 3. Lazy Initialization

Models loaded on first use, not at import:
- Fast startup for health checks
- Resources allocated only when needed

### 4. No Hardcoded Values

All configuration via Pydantic models:
- Environment-driven
- Type-safe
- Validated at startup

### 5. Citation Format

Citations are segment-based:

```python
class Citation:
    chunk_id: ChunkId
    text: str           # Source text
    source_document: str # Filename or ID
    page: int | None    # Page number if available
```

Adapter pattern allows other formats if needed.

## Testing Strategy

### Unit Tests
- Domain logic with no mocks
- Pure Python assertions

### Contract Tests
- Verify adapter fulfills Protocol
- Runtime type checking

### Integration Tests
- Real adapters with local services (embedded Chroma)
- No external API calls

### E2E Tests
- Full pipeline with mock adapters
- No API keys required

## Telemetry

OpenTelemetry spans for observability:

```python
from bhodi_platform.infrastructure.tracing import traced

@traced("custom.operation")
async def my_operation(self):
    ...
```

Spans include:
- Operation name
- Provider/model attributes
- Duration

## Extensibility

Adding a new adapter:

1. Create adapter in `infrastructure/<adapter_type>/`
2. Implement the port Protocol
3. Register in `Container._create_<adapter_type>_adapter()`
4. Add tests

No modification to domain or application layers required.
