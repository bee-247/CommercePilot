"""Customer-service RAG chat application service."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

from fastapi import HTTPException

from orchestrator.legacy_rag_agent import chat_with_agent, chat_with_agent_stream
from schemas.rag_schemas import ChatRequest, ChatResponse


class RagChatService:
    """Owns RAG chat orchestration and API-friendly error mapping."""

    def chat(self, *, request: ChatRequest, username: str) -> ChatResponse:
        try:
            session_id = request.session_id or "default_session"
            response = chat_with_agent(request.message, username, session_id)
            if isinstance(response, dict):
                return ChatResponse(**response)
            return ChatResponse(response=response)
        except Exception as exc:
            self._raise_model_error(exc)

    async def chat_stream(
        self,
        *,
        request: ChatRequest,
        username: str,
    ) -> AsyncIterator[str]:
        try:
            session_id = request.session_id or "default_session"
            async for chunk in chat_with_agent_stream(
                request.message,
                username,
                session_id,
            ):
                yield chunk
        except Exception as exc:
            error_data = {"type": "error", "content": str(exc)}
            yield f"data: {json.dumps(error_data)}\n\n"

    def _raise_model_error(self, exc: Exception) -> None:
        message = str(exc)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if not match:
            raise HTTPException(status_code=500, detail=message) from exc

        code = int(match.group(1))
        if code == 429:
            raise HTTPException(
                status_code=429,
                detail=(
                    "上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n"
                    f"原始错误：{message}"
                ),
            ) from exc
        if code in (401, 403):
            raise HTTPException(status_code=code, detail=message) from exc
        raise HTTPException(status_code=code, detail=message) from exc
