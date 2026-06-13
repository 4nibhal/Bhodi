"""
Retrieval adapters.

Empty: the retrieval context has no context-local adapters. The
underlying adapter implementations (embedding providers, vector
store backends, rerankers) live in
`bodhi_rag.infrastructure.{embedding,vector_store,reranker}/`
and are injected into `RetrieveQueryUseCase` as the cross-context
ports.

If a context-local adapter emerges (e.g. a metadata-filtered
vector store wrapper, or a hybrid-search orchestrator), it
belongs here.
"""
