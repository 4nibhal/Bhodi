"""
Conversation memory use cases.

The application-level entry points into the conversation bounded
context. The `BhodiApplication` facade depends on these, not on the
port directly, so that cross-cutting concerns (logging, telemetry,
authorization) can be added here without leaking into adapters.
"""
