"""
Indexing use cases.

The application-layer entry points for the indexing bounded context.
Currently the public use cases are `IndexDocumentUseCase` (parse
-> chunk -> embed -> add) and `DeleteDocumentUseCase` (vector
store delete). Both delegate to the cross-context ports; if a
context-local port emerges (e.g. a metadata-aware indexer that
filters by document type), it belongs in `indexing/ports/`.
"""
