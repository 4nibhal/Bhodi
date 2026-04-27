"""Query endpoint."""

from fastapi import APIRouter, HTTPException, Query, status

from bhodi_platform.application.models import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Query the indexed documents.

    Ask a question and get an answer with citations.
    """
    from bhodi_platform.interfaces.api.app import get_bhodi_app

    app = get_bhodi_app()
    try:
        return await app.query(request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/conversations/{conversation_id}", status_code=status.HTTP_200_OK)
async def get_conversation(conversation_id: str) -> dict:
    """
    Get conversation history.

    Returns all turns in a conversation.
    """
    from bhodi_platform.interfaces.api.app import get_bhodi_app
    from bhodi_platform.domain.value_objects import ConversationId

    app = get_bhodi_app()
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
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
