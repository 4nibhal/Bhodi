"""
Conversation memory adapters.

Concrete implementations of `ConversationMemoryPort`. The volatile
(in-process) adapter is the default; persistent adapters (SQLite,
PostgreSQL) are out of scope for F5 and tracked separately.
"""
