"""
Conversation memory domain.

Empty for now: the only entity (ConversationTurn) is shared with the
retrieval flow and lives in the top-level `domain/` package. When
the conversation context grows its own invariants (e.g. summarization
windows, persistence policies, multi-tenant scoping), the entities
and value objects that express them belong here.
"""
