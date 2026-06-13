"""
Retrieval ports.

Empty: the retrieval context uses the cross-context ports
(EmbeddingPort, VectorStorePort, RerankerPort) re-imported from
`bodhi_rag.ports.*`. If a context-local port emerges (e.g. a
`RelevanceScorer` that combines vector similarity with metadata
filters, or a `HybridSearch` port that mixes vector and keyword
search), it belongs here.
"""
