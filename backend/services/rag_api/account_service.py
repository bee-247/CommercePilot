"""Account and RAG session application service."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from core.auth import authenticate_user, create_access_token, get_password_hash
from orchestrator.legacy_rag_agent import storage
from rag.storage.models import User
from rag.utils.token_usage_tracker import (
    delete_session_token_usage,
    get_session_token_usage,
)
from schemas.rag_schemas import (
    AuthResponse,
    CurrentUserResponse,
    LoginRequest,
    MessageInfo,
    RegisterRequest,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
    TokenUsageResponse,
)


class RagAccountService:
    """Owns auth/session behavior so API handlers stay thin."""

    def register(self, request: RegisterRequest, db: Session) -> AuthResponse:
        username = (request.username or "").strip()
        password = (request.password or "").strip()
        if not username or not password:
            raise HTTPException(status_code=400, detail="用户名和密码不能为空")

        exists = db.query(User).filter(User.username == username).first()
        if exists:
            raise HTTPException(status_code=409, detail="用户名已存在")

        role = "user"
        user = User(
            username=username,
            password_hash=get_password_hash(password),
            role=role,
        )
        db.add(user)
        db.commit()

        token = create_access_token(username=username, role=role)
        return AuthResponse(access_token=token, username=username, role=role)

    def login(self, request: LoginRequest, db: Session) -> AuthResponse:
        user = authenticate_user(db, request.username, request.password)
        if not user:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        token = create_access_token(username=user.username, role=user.role)
        return AuthResponse(
            access_token=token,
            username=user.username,
            role=user.role,
        )

    def current_user(self, user: User) -> CurrentUserResponse:
        return CurrentUserResponse(username=user.username, role=user.role)

    def session_messages(
        self,
        *,
        username: str,
        session_id: str,
    ) -> SessionMessagesResponse:
        try:
            messages = [
                MessageInfo(
                    type=msg["type"],
                    content=msg["content"],
                    timestamp=msg["timestamp"],
                    rag_trace=msg.get("rag_trace"),
                )
                for msg in storage.get_session_messages(username, session_id)
            ]
            return SessionMessagesResponse(messages=messages)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def list_sessions(self, *, username: str) -> SessionListResponse:
        try:
            sessions = [
                SessionInfo(**item)
                for item in storage.list_session_infos(username)
            ]
            sessions.sort(key=lambda item: item.updated_at, reverse=True)
            return SessionListResponse(sessions=sessions)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def delete_session(
        self,
        *,
        username: str,
        session_id: str,
    ) -> SessionDeleteResponse:
        try:
            deleted = storage.delete_session(username, session_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="会话不存在")
            delete_session_token_usage(username, session_id)
            return SessionDeleteResponse(
                session_id=session_id,
                message="成功删除会话",
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def token_usage(self, *, username: str, session_id: str) -> TokenUsageResponse:
        usage = get_session_token_usage(username, session_id)
        return TokenUsageResponse(session_id=session_id, **usage)
