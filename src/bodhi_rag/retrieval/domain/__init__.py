"""
Retrieval domain.

Empty for now: `RetrievedDocument` is the only entity and it is
shared with the indexing flow (the in-memory vector store returns
it on `search`, the indexing flow constructs it on `add`). It
lives in the top-level `domain/` package for that reason. If the
retrieval context grows its own invariants (relevance scoring,
diversity, recency), the entities and value objects that express
them belong here.
"""
