"""
Answer synthesis adapters.

Empty: the answering context has no context-local adapters. The
LLM providers (openai, ollama, mock) live in
`bodhi_rag.infrastructure.llm` and are injected into
`SynthesizeAnswerUseCase` as the cross-context `LLMPort`.

If a context-local adapter emerges (e.g. a streaming OpenAI client
with answer-streaming semantics specific to bodhi-rag's UX), it
belongs here.
"""
