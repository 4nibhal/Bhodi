from __future__ import annotations

from typing import Any

from bhodi_platform.conversation.settings import ConversationSettings


def _vectorstore_class() -> Any:
    from langchain_chroma import Chroma

    return Chroma


def build_vectorstore(settings: ConversationSettings, embeddings: Any) -> Any:
    return _vectorstore_class()(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.persist_directory),
    )


def build_retriever(
    vectorstore: Any,
    settings: ConversationSettings,
    *,
    conversation_id: str | None,
) -> Any:
    search_kwargs: dict[str, Any] = {"k": settings.retriever_k}
    if conversation_id is not None:
        search_kwargs["filter"] = {"conversation_id": conversation_id}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
