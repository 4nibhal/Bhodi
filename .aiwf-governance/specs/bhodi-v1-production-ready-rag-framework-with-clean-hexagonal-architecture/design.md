# Design: Bhodi v1 - Production-Ready RAG Framework

## Context
- **Spec ID**: bhodi-v1-production-ready-rag-framework-with-clean-hexagonal-architecture
- **Created**: 2026-04-04
- **Status**: draft
- **Approach selected**: Hexagonal architecture with constructor-based DI, designed for future plugin extensibility

## Problem Statement

Bhodi tiene potencial como framework RAG de calidad, pero actualmente adolece de:

1. **Calidad de código inconsistente**: Estructura hexagonal existe pero faltan contracts formales, type hints parciales, error handling incompleto
2. **Testing insuficiente**: 151 tests existen pero falta cobertura E2E real y regression gates significativas
3. **Documentación fragmentada**: README existe pero docs/ está vacío, sin API docs generados ni arch diagrams actualizados
4. **Naming/confusión de marca**: Package name "bhodi-doc-analyzer" no refleja el producto
5. **Legacy shims activos**: bhodi_doc_analyzer/ e indexer/ aún presentes cuando deberían estar extraídos o deprecados
6. **Extensibilidad limitada**: No hay forma limpia de añadir nuevos chunkers, embedders, o vector stores sin modificar código core

El objetivo no es añadir features - es elevar la calidad del existente a nivel comparable con LlamaIndex/Haystack para facilitar adopción técnica.

## Proposed Solution

Construir Bhodi como un **framework RAG horizontal de calidad production-ready**:

- Arquitectura hexagonal con DI simple via constructor
- Ports abstractos bien definidos (EmbeddingPort, VectorStorePort, ChunkingPort, LLM answering port)
- Implementaciones concretas pluggables via config
- Código de calidad ingeniero senior (type hints completos, error handling robusto, zero TODOs)
- Tests que rivalicen con frameworks establecidos
- Documentación generated from code + arch decision records
- Diseñado para que un eventual plugin system sea natural, no forzado

El producto es el **framework** - cómo se monetiza (hosting, enterprise, etc.) es preocupación separada.

## Architecture

### Principles

1. **Hexagonal (Ports & Adapters)**: Domain es el centro, interfaces son ports, implementaciones son adapters
2. **Constructor Injection**: Dependencias pasadas via constructor, no globals ni singletons mágicos
3. **Protocol-based contracts**: Usar Protocol de Python para ports (no ABCs rígidos), permite structural typing
4. **Config-driven runtime**: Todo comportamiento configurable via environment/config, zero hardcoding
5. **Fail loud**: Errors descriptivos, nunca silencios inesperados

### Components

```
bhodi_platform/
├── domain/                    # Pure business logic, zero external deps
│   ├── entities.py           # Query, Document, Chunk, Answer, ConversationTurn
│   ├── value_objects.py       # EmbeddingVector, DocumentId, ChunkId, etc.
│   ├── services.py           # Domain services (e.g., chunking strategy logic)
│   └── exceptions.py          # Domain exceptions (DocumentNotFound, etc.)
│
├── application/              # Use cases, orchestration
│   ├── facade.py             # BhodiApplication - main entry point
│   ├── indexing.py           # IndexDocumentUseCase
│   ├── query.py              # QueryAnswerUseCase
│   └── models.py             # Request/Response models (no domain, no infra)
│
├── ports/                    # Abstract interfaces (Protocols)
│   ├── embedding.py           # EmbeddingPort (generate, embed_documents, embed_query)
│   ├── vector_store.py        # VectorStorePort (add, search, delete, persist)
│   ├── document_parser.py     # DocumentParserPort (parse, extract_text, extract_metadata)
│   ├── chunker.py            # ChunkerPort (chunk, chunk_size, overlap)
│   ├── llm.py                # LLMPort (generate, generate_with_context)
│   └── conversation_memory.py # ConversationMemoryPort (add, get_history, clear)
│
├── infrastructure/          # Concrete adapters, DI composition
│   ├── composition.py         # Container - wires ports to implementations
│   ├── embedding/
│   │   ├── openai.py         # OpenAIEmbeddings adapter
│   │   ├── local.py          # sentence-transformers adapter
│   │   └── cohere.py         # Cohere adapter
│   ├── vector_store/
│   │   ├── chroma.py         # Chroma adapter
│   │   ├── qdrant.py         # Qdrant adapter
│   │   └── in_memory.py      # Simple in-memory for testing
│   ├── document_parser/
│   │   ├── pypdf.py          # PyPDF adapter
│   │   └── unstructured.py   # Unstructured.io adapter
│   ├── chunker/
│   │   ├── fixed_size.py     # Fixed-size chunker
│   │   ├── recursive.py      # Recursive character splitter
│   │   └── semantic.py       # Semantic chunker (future)
│   ├── llm/
│   │   ├── openai.py         # OpenAI adapter
│   │   ├── anthropic.py      # Anthropic adapter
│   │   └── ollama.py         # Ollama local adapter
│   └── conversation_memory/
│       ├── volatile.py       # In-memory conversation store
│       └── persistent.py     # File-based persistent store
│
├── interfaces/              # Transport adapters (API, CLI, TUI)
│   ├── api/
│   │   ├── server.py         # FastAPI app
│   │   ├── routes/
│   │   │   ├── health.py     # Health endpoint (no model init)
│   │   │   ├── indexing.py   # POST /documents, DELETE /documents/{id}
│   │   │   └── query.py      # POST /query, GET /conversations/{id}
│   │   └── middleware.py     # Error handling, logging, cors
│   ├── cli/
│   │   ├── indexing.py       # bhodi-index CLI commands
│   │   └── query.py          # bhodi query CLI
│   └── worker/
│       └── task_queue.py     # Background job adapter
│
└── evaluation/               # Testing, metrics
    ├── fixtures.py           # Test fixtures
    ├── metrics.py            # Retrieval metrics (precision@k, recall@k)
    └── thresholds.py          # Regression gates
```

### Data Flow

**Indexing Pipeline:**
```
User Upload (PDF/TXT)
    → DocumentParserPort.parse()
    → Document (domain entity with text + metadata)
    → ChunkerPort.chunk()
    → List[Chunk] (domain entities)
    → EmbeddingPort.embed_documents()
    → List[Tuple[Chunk, EmbeddingVector]]
    → VectorStorePort.add()
    → persisted index
```

**Query Pipeline:**
```
User Query String
    → EmbeddingPort.embed_query()
    → EmbeddingVector
    → VectorStorePort.search(query_embedding, top_k)
    → List[RetrievedDocument] (ranked by similarity)
    → LLMPort.generate_with_context(query, retrieved_docs, conversation_history)
    → Answer (with citations)
    → ConversationMemoryPort.add()
    → Response to user
```

### API Contract

**Core Facade (application/facade.py):**
```python
class BhodiApplication:
    def __init__(
        self,
        document_parser: DocumentParserPort,
        chunker: ChunkerPort,
        embedding: EmbeddingPort,
        vector_store: VectorStorePort,
        llm: LLMPort,
        conversation_memory: ConversationMemoryPort,
    ) -> None:
        ...

    async def index_document(
        self,
        source: Union[Path, bytes, BinaryIO],
        metadata: Optional[dict] = None,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> DocumentId:
        """Parse, chunk, embed, and store a document."""
        ...

    async def query(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
        temperature: float = 0.7,
    ) -> Answer:
        """Query the index and return an answer with citations."""
        ...

    async def health_check(self) -> HealthStatus:
        """Check if core services are initialized (no model loading)."""
        ...
```

**Ports (ports/*.py):**
```python
class EmbeddingPort(Protocol):
    """Protocol for embedding generation."""
    async def embed_documents(
        self, texts: list[str]
    ) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...
    async def dimensions(self) -> int: ...

class VectorStorePort(Protocol):
    """Protocol for vector storage and retrieval."""
    async def add(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None: ...
    async def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievedDocument]: ...
    async def delete(self, document_id: DocumentId) -> None: ...
    async def persist(self) -> None: ...
```

### Configuration

All runtime configuration via environment or config file:

```yaml
# config.yaml (or env vars)
bhodi:
  # Document parsing
  parser:
    provider: pypdf  # or "unstructured"

  # Chunking
  chunker:
    provider: recursive
    chunk_size: 512
    overlap: 64

  # Embeddings
  embedding:
    provider: openai  # or "local", "cohere"
    model: text-embedding-3-small
    dimensions: 1536

  # Vector store
  vector_store:
    provider: chroma  # or "qdrant", "in_memory"
    persist_directory: ./data/chroma

  # LLM for answering
  llm:
    provider: openai  # or "anthropic", "ollama"
    model: gpt-4o-mini
    temperature: 0.7

  # Conversation memory
  conversation:
    provider: volatile  # or "persistent"
```

## Constraints

### Technical Constraints
- Python 3.11+ only (using newer typing features)
- Type hints required on all public interfaces (no `Any` in contracts)
- Async-first (all ports async), sync adapters wrap sync libs
- No import-time side effects (model loading, filesystem writes, network calls)
- Lazy initialization (models loaded on first use, not at import)

### Performance Requirements
- Health endpoint < 50ms (no model loading)
- Embedding generation: configurable batch size
- Vector search: < 100ms for 10K documents on local Chroma
- Graceful degradation if optional deps missing

### Quality Requirements
- Zero TODO/FIXME in production code
- Zero bare `except:` clauses
- All exceptions logged with context
- Test coverage > 80% on core (application, domain)
- Type checking via pyright strict mode

## Alternatives Considered

### Option A: Plugin System from Day 1
- **Description**: Full plugin architecture with entry points, discovery, and sandboxing
- **Pros**: Maximum flexibility, community extensibility
- **Cons**: Over-engineering for v1, slows down core development, complex testing
- **Verdict**: Rejected for v1. Design to make it possible later (Option B is compatible)

### Option B: Strict ABC with Template Method
- **Description**: Abstract base classes with template methods for lifecycle
- **Pros**: Strong contracts, IDE autocomplete
- **Cons**: Python's ABCs are rigid; breaking changes in ABC cascade widely
- **Verdict**: Rejected. Protocol + structural typing is more flexible

### Option C: Current State (Incremental)
- **Description**: Keep current structure, fix incrementally
- **Pros**: No big bang refactor
- **Cons**: Technical debt accumulates; quality bar stays inconsistent
- **Verdict**: Rejected. User wants honest, solid foundation

### Option D: LlamaIndex as Core (fork/extend)
- **Description**: Fork LlamaIndex, strip to core, build around it
- **Pros**: Leverage existing battle-tested code
- **Cons**: Dependency on LlamaIndex release cycle; their decisions become yours
- **Verdict**: Rejected. Bhodi should be independent, can integrate with LlamaIndex as adapter

## Testing Strategy

### Unit Tests
- Domain entities: all methods, edge cases
- Value objects: validation, equality, serialization
- Domain services: chunking logic, policy enforcement
- Application use cases: happy path + error paths mocked

### Integration Tests
- Each adapter with real implementation (requires service running)
- Adapter composition via Container
- Config loading and validation

### Contract Tests
- Each port implementation validates against Protocol
- Generated test stubs for new implementations

### E2E Tests
- Full indexing + query flow via API
- CLI end-to-end
- Health check at each stage

### Evaluation/Regression
- Retrieval metrics on fixed dataset
- Citation accuracy checks
- Latency benchmarks

### Test Structure
```
tests/
├── unit/
│   └── bhodi_platform/
│       ├── domain/
│       ├── application/
│       └── ports/
├── integration/
│   ├── adapters/
│   └── api/
├── contract/
│   └── test_port_contracts.py
└── e2e/
    └── test_full_pipeline.py
```

## Out of Scope

### Intentionally Excluded from v1
- **Multi-tenant isolation**: Single API key per instance in v1
- **SSO/SAML/OAuth**: API keys only
- **Dashboard/UI**: API-first product
- **Plugin system**: Designed to be added later
- **Multi-format support beyond PDF/TXT**: DOCX, HTML, Markdown as future
- **Fine-tuning**: Not in core
- **GraphRAG / advanced retrieval**: Roadmap, not v1
- **Cloud hosting automation**: Terraform/Ansible charts
- **Billing/payment integration**: Separate concern

## Open Questions

1. **Citation format**: Still use current approach (segment + page) or adopt a standard (Anthropic citations, etc.)?
2. **Conversation memory granularity**: Per-user vs per-session vs global?
3. **Error taxonomy**: Should we define a unified error codesystem (similar to HTTP but for RAG)?
4. **Telemetry**: Add basicopentelemetry or defer to operator to instrument?

---

## Spec Acceptance Criteria

1. [ ] All ports have Protocol definitions with complete type signatures
2. [ ] All adapters implement corresponding ports
3. [ ] Container wires implementations based on config
4. [ ] Health endpoint returns 200 without loading models
5. [ ] Full indexing pipeline works: PDF → embedded chunks → stored
6. [ ] Full query pipeline works: question → embedded → retrieved → answered with citations
7. [ ] Zero TODOs/FIXMEs in production code
8. [ ] pyright strict passes on bhodi_platform/
9. [ ] >80% coverage on application and domain layers
10. [ ] API docs generated from code (Sphinx or MkDocs)
11. [ ] Legacy shims (bhodi_doc_analyzer, indexer) extracted to separate compat package
12. [ ] Package renamed to `bhodi` (not `bhodi-doc-analyzer`)
