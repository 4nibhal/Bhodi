"""Query endpoint."""

from fastapi import APIRouter, HTTPException, status

from bodhi_rag.application.models import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Query the indexed documents.

    Ask a question and get an answer with citations.
    """
    from bodhi_rag.interfaces.api.app import get_bodhi_rag_app

    app = get_bodhi_rag_app()
    try:
        return await app.query(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from err


@router.get("/conversations/{conversation_id}", status_code=status.HTTP_200_OK)
async def get_conversation(conversation_id: str) -> dict:
    """
    Get conversation history.

    Returns all turns in a conversation.
    """
    from bodhi_rag.domain.value_objects import ConversationId
    from bodhi_rag.interfaces.api.app import get_bodhi_rag_app

    app = get_bodhi_rag_app()
    try:
        conv_id = ConversationId(conversation_id)
        history = await app.get_conversation_history(conv_id)
        return {
            "conversation_id": conversation_id,
            "turns": [
                {
                    "user_message": turn.user_message,
                    "assistant_message": turn.assistant_message,
                    "turn_index": turn.turn_index,
                }
                for turn in history
            ],
        }
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from err
