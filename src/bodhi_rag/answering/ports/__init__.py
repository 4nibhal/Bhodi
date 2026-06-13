"""
Answer synthesis ports.

Empty: the answering use case currently depends directly on the
cross-context `LLMPort`. If a context-local `AnswerSynthesizerPort`
emerges (e.g. to abstract over multiple LLM-as-answerer strategies
or to add refusal/grounding checks at the port boundary), it
belongs here.
"""
