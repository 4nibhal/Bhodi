"""
Retrieval bounded context.

Owns the responsibility of turning a user question into a
relevance-ranked list of candidate chunks. The query pipeline
this context encapsulates is:

    embed(question)  →  vector_store.search(embedding, top_k * overfetch)
                    →  reranker.rerank(question, candidates, top_k)

The orchestration is the `RetrieveQueryUseCase`; the underlying
abstractions (EmbeddingPort, VectorStorePort, RerankerPort) are
cross-context ports re-imported here.

Hexagonal layout:

    retrieval/
    ├── domain/         (empty for now; RetrievedDocument is the
    │                   only entity and it lives in top-level domain/)
    ├── application/retrieve.py  (RetrieveQueryUseCase)
    ├── ports/          (empty: the context uses the cross-context
    │                   ports; if a context-local port emerges it
    │                   belongs here)
    └── infrastructure/ (empty: no context-local adapters; the
                         adapter implementations live in
                         infrastructure/{embedding,vector_store,reranker}/)
"""
