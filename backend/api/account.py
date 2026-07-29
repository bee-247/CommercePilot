"""Authentication and RAG session route group."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.auth import get_current_user, get_db
from rag.storage.models import User
from schemas.rag_schemas import (
    AuthResponse,
    CurrentUserResponse,
    LoginRequest,
    RegisterRequest,
    SessionDeleteResponse,
    SessionListResponse,
    SessionMessagesResponse,
    TokenUsageResponse,
)
from services.rag_api.account_service import RagAccountService


router = APIRouter(tags=["account"])
account_service = RagAccountService()


@router.post("/auth/register", response_model=AuthResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    return account_service.register(request, db)


@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    return account_service.login(request, db)


@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return account_service.current_user(current_user)


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    return account_service.session_messages(
        username=current_user.username,
        session_id=session_id,
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: User = Depends(get_current_user)):
    return account_service.list_sessions(username=current_user.username)


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    return account_service.delete_session(
        username=current_user.username,
        session_id=session_id,
    )


@router.get("/sessions/{session_id}/token-usage", response_model=TokenUsageResponse)
async def get_session_token_usage_endpoint(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    return account_service.token_usage(
        username=current_user.username,
        session_id=session_id,
    )
