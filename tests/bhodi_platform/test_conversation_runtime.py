from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from bhodi_platform.conversation.runtime import (
    get_persistent_retriever,
    initialize_persistent_runtime,
)
from bhodi_platform.conversation.settings import ConversationSettings


class ConversationRuntimeTest(TestCase):
    def test_initialize_runtime_uses_dedicated_settings(self) -> None:
        vectorstore = object()
        settings = ConversationSettings(
            persist_directory=Path("/tmp/chroma-conversations"),
            collection_name="conversation-memory",
            retriever_k=5,
        )

        with patch(
            "bhodi_platform.conversation.runtime.build_vectorstore",
            return_value=vectorstore,
        ) as build_vectorstore:
            returned = initialize_persistent_runtime(
                settings=settings,
                embeddings_factory=lambda: "embeddings",
            )

        self.assertIs(returned, vectorstore)
        self.assertEqual(build_vectorstore.call_args.args[0], settings)
        self.assertEqual(build_vectorstore.call_args.args[1], "embeddings")

    def test_get_persistent_retriever_filters_by_conversation_id(self) -> None:
        settings = ConversationSettings(
            persist_directory=Path("/tmp/chroma-conversations"),
            retriever_k=4,
        )

        with patch(
            "bhodi_platform.conversation.runtime.ConversationSettings.from_environment",
            return_value=settings,
        ):
            with patch(
                "bhodi_platform.conversation.runtime.get_persistent_vectorstore",
                return_value="vectorstore",
            ):
                with patch(
                    "bhodi_platform.conversation.runtime.build_retriever",
                    return_value="retriever",
                ) as build_retriever:
                    returned = get_persistent_retriever("conv-123")

        self.assertEqual(returned, "retriever")
        self.assertEqual(build_retriever.call_args.args[0], "vectorstore")
        self.assertEqual(build_retriever.call_args.args[1], settings)
        self.assertEqual(
            build_retriever.call_args.kwargs["conversation_id"],
            "conv-123",
        )
