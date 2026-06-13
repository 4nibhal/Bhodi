"""
Conversation memory bounded context.

Owns the responsibility of persisting and retrieving per-conversation
turn history. The hexagonal layout is:

    conversation/
    ├── domain/        (entities specific to the context; here, empty —
    │                   ConversationTurn stays in the top-level domain/
    │                   because it is shared with the retrieval flow)
    ├── application/   (use cases: ConversationMemoryUseCase)
    ├── ports/         (Protocol: ConversationMemoryPort)
    └── infrastructure/ (adapters: VolatileConversationMemoryAdapter)

The public API of the bounded context is exported from this __init__.
Consumers (the application facade, other bounded contexts) should
import only from `bodhi_rag.conversation`, never from the
sub-modules.
"""
