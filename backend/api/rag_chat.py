"""Customer-service RAG chat route group."""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from core.auth import get_current_user
from rag.storage.models import User
from schemas.rag_schemas import ChatRequest, ChatResponse
from services.rag_api.rag_chat_service import RagChatService


router = APIRouter(tags=["rag-chat"])
rag_chat_service = RagChatService()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    return rag_chat_service.chat(request=request, username=current_user.username)


@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    return StreamingResponse(
        rag_chat_service.chat_stream(
            request=request,
            username=current_user.username,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
