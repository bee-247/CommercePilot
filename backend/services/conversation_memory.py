from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from core.model_clients import create_chat_model

from core.config import get_settings
from agents.memory_update_agent import MemoryUpdateAgent
from database.models import (
    ConversationSessionRecord,
    MemoryObservationRecord,
    UserLongTermMemoryRecord,
)
from database.session import UserSessionLocal
from models.schemas import ChatHistoryMessage, RecommendationResponse
from services.token_counter import estimate_tokens
from utils.json_utils import parse_json_object


LONG_MEMORY_TTL_DAYS = {
    "preferred_category": 60,
    "disliked_category": 60,
    "style_tag": 60,
    "price_preference": 30,
    "usage_scene": 30,
    "other": 30,
}


SUMMARY_PROMPT = """你是电商导购会话记忆整理Agent。你只输出JSON，不要解释。

请把最近几轮用户对话、工具/Agent输出和最终回答压缩成一段短期记忆摘要，用于后续多轮推荐召回。

要求:
1. 用自然语言保留仍然有效的购物需求、预算、类目、排除条件、图片语义和用户追问约束。
2. 可以概括最近推荐过的商品和用户明确接受/拒绝/偏好的信息。
3. 不要把一次性闲聊写成偏好。
4. 不要编造用户没有表达过的信息。

输出格式:
{
  "summary": "简短会话摘要"
}
"""


LONG_TERM_PROMPT = """你是电商长期记忆抽取Agent。你只输出JSON，不要解释。

请从整个会话短期记忆中判断是否有值得写入用户长期记忆的信息。

只抽取稳定偏好，例如长期喜欢的类目、风格、价格带、明确排斥项、常用场景。
不要抽取一次性需求，例如“今天想买”“这次预算”“这张图类似款”，除非用户明确说这是长期偏好。
不要抽取敏感个人信息。

输出格式:
{
  "should_write": true,
  "memories": [
    {
      "type": "preferred_category" | "style_tag" | "price_preference" | "disliked_category" | "usage_scene" | "other",
      "value": "记忆内容",
      "confidence": 0.0,
      "evidence": "来自会话的简短证据"
    }
  ],
  "reason": "简短原因"
}
"""


@dataclass
class SessionMemory:
    user_id: str
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turn_count: int = 0
    summary: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    long_term_candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_context(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "summary": self.summary,
            "recent_messages": self.messages[-8:],
            "recent_turns": self.turns[-4:],
        }


class ConversationMemoryStore:
    """Database-backed session memory with LLM compaction hooks."""

    def __init__(
        self,
        summarize_every: int = 8,
        max_messages: int = 16,
        max_memory_tokens: int | None = None,
    ):
        settings = get_settings()
        self.summarize_every = summarize_every
        self.max_messages = max_messages
        self.max_memory_tokens = (
            max_memory_tokens
            if max_memory_tokens is not None
            else settings.session_memory_max_tokens
        )
        self.llm = create_chat_model(
            temperature=0.0,
            max_tokens=900,
        )
        self.memory_update_agent = MemoryUpdateAgent()

    def session_key(self, user_id: str, session_id: str | None = None) -> str:
        clean_user_id = user_id or "web_user"
        clean_session_id = session_id or clean_user_id
        return f"{clean_user_id}:{clean_session_id}"

    def get(self, user_id: str, session_id: str | None = None) -> SessionMemory:
        key = self.session_key(user_id, session_id)
        with UserSessionLocal() as session:
            record = session.get(ConversationSessionRecord, key)
            if record:
                return self._record_to_memory(record)

            memory = SessionMemory(
                user_id=user_id or "web_user",
                session_id=session_id or user_id or "web_user",
            )
            session.add(self._memory_to_record(memory))
            session.commit()
            return memory

    def history_for_agent(
        self,
        memory: SessionMemory,
    ) -> list[Any]:
        return [
            ChatHistoryMessage(
                role=str(item.get("role") or ""),
                content=str(item.get("content") or ""),
            )
            for item in memory.messages[-8:]
            if item.get("role") and item.get("content")
        ]

    def list_sessions(
        self,
        user_id: str,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        clean_user_id = user_id or "web_user"
        with UserSessionLocal() as session:
            records = (
                session.query(ConversationSessionRecord)
                .filter(ConversationSessionRecord.user_id == clean_user_id)
                .order_by(ConversationSessionRecord.updated_at.desc())
                .limit(max(1, min(limit, 100)))
                .all()
            )
        sessions = []
        for record in records:
            messages = self._dict_list(self._loads(record.messages_json, []))
            if not messages:
                continue
            preview = next(
                (
                    str(item.get("content") or "")
                    for item in reversed(messages)
                    if item.get("role") == "user"
                ),
                "新对话",
            )
            sessions.append(
                {
                    "session_id": record.session_id,
                    "updated_at": record.updated_at.isoformat()
                    if record.updated_at
                    else "",
                    "message_count": len(messages),
                    "preview": preview[:60],
                }
            )
        return sessions

    def session_messages(
        self,
        user_id: str,
        session_id: str,
    ) -> list[dict[str, str]]:
        key = self.session_key(user_id, session_id)
        with UserSessionLocal() as session:
            record = session.get(ConversationSessionRecord, key)
            if record is None or record.user_id != (user_id or "web_user"):
                return []
            messages = self._dict_list(self._loads(record.messages_json, []))
        return [
            {
                "role": str(item.get("role") or ""),
                "content": str(item.get("content") or ""),
            }
            for item in messages
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]

    def delete_session(self, user_id: str, session_id: str) -> bool:
        clean_user_id = user_id or "web_user"
        key = self.session_key(clean_user_id, session_id)
        with UserSessionLocal() as session:
            record = session.get(ConversationSessionRecord, key)
            if record is None or record.user_id != clean_user_id:
                return False
            session.delete(record)
            session.commit()
        return True

    async def update(
        self,
        memory: SessionMemory,
        user_message: str,
        assistant_message: str,
        context: dict[str, Any],
        response: RecommendationResponse,
    ) -> None:
        tool_events = self._tool_events(response)

        memory.turn_count += 1
        memory.updated_at = time.time()
        memory.messages.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        memory.messages = memory.messages[-self.max_messages :]
        recommended_product_ids = [product.product_id for product in response.products]
        memory.turns.append(
            {
                "user_message": user_message,
                "assistant_message": assistant_message,
                "context": self._compact_context(context),
                "tool_events": tool_events,
                "recommended_product_ids": recommended_product_ids,
            }
        )
        memory.turns = memory.turns[-self.summarize_every :]

        if self._should_summarize(memory):
            data = await self.summarize(memory)
            if not data:
                self._save(memory)
        else:
            self._save(memory)

    async def summarize(self, memory: SessionMemory) -> dict[str, Any]:
        payload = {
            "previous_summary": memory.summary,
            "recent_turns": memory.turns,
            "current_memory": memory.to_context(),
        }
        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=SUMMARY_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            )
            data = parse_json_object(response.content)
        except Exception:
            return {}

        memory.summary = str(data.get("summary") or memory.summary or "")
        memory.turns = []
        self._save(memory)
        return data

    async def end_session(
        self,
        user_id: str,
        session_id: str | None = None,
        clear: bool = True,
    ) -> dict[str, Any]:
        key = self.session_key(user_id, session_id)
        with UserSessionLocal() as session:
            record = session.get(ConversationSessionRecord, key)
            memory = self._record_to_memory(record) if record else None
        if not memory:
            return {"session_id": session_id or user_id, "memories": []}

        if memory.turns:
            await self.summarize(memory)

        payload = {
            "session_memory": memory.to_context(),
            "messages": memory.messages,
        }
        data = await self.memory_update_agent.extract_memories(**payload)
        clean_memories = self._clean_memory_candidates(data.get("memories"))
        memory.long_term_candidates = clean_memories
        saved_memories = await self._apply_long_term_memories(
            memory=memory,
            memories=clean_memories,
            raw_text=memory.summary,
            source="session_end",
        )
        if clear:
            self._delete_session(key)
        else:
            self._save(memory)
        return {
            "session_id": memory.session_id,
            "user_id": memory.user_id,
            "memories": saved_memories,
            "reason": str(data.get("reason") or ""),
        }

    async def update_long_term_from_message(
        self,
        memory: SessionMemory,
        user_message: str,
    ) -> list[dict[str, Any]]:
        data = await self.memory_update_agent.extract_memories(
            message=user_message,
            session_memory=memory.to_context(),
            messages=memory.messages[-8:],
        )
        memories = self._clean_memory_candidates(data.get("memories"))
        if not memories:
            return []
        saved = await self._apply_long_term_memories(
            memory=memory,
            memories=memories,
            raw_text=user_message,
            source="preference_update",
        )
        memory.long_term_candidates = saved
        self._save(memory)
        return saved

    def long_term_candidates(
        self,
        user_id: str,
        limit: int = 20,
        memory_types: list[str] | None = None,
        query: str = "",
    ) -> list[dict[str, Any]]:
        self._expire_active_long_term_memories(user_id)
        now = datetime.utcnow()
        with UserSessionLocal() as session:
            stmt = session.query(UserLongTermMemoryRecord).filter(
                UserLongTermMemoryRecord.user_id == user_id
            )
            stmt = stmt.filter(UserLongTermMemoryRecord.state == "active")
            stmt = stmt.filter(
                (UserLongTermMemoryRecord.expires_at.is_(None))
                | (UserLongTermMemoryRecord.expires_at > now)
            )
            if memory_types:
                stmt = stmt.filter(UserLongTermMemoryRecord.memory_type.in_(memory_types))
            rows = (
                stmt.order_by(UserLongTermMemoryRecord.created_at.desc())
                .limit(limit * 3 if query else limit)
                .all()
            )
        if query:
            rows = self._rank_long_term_rows(rows, query)[:limit]
        return [
            {
                "id": row.id,
                "type": row.memory_type,
                "memory_key": self._row_memory_key(row),
                "state": row.state,
                "value": row.value,
                "confidence": row.confidence,
                "evidence": row.evidence,
                "session_id": row.session_id,
            }
            for row in rows
        ]

    def _rank_long_term_rows(
        self,
        rows: list[UserLongTermMemoryRecord],
        query: str,
    ) -> list[UserLongTermMemoryRecord]:
        tokens = set(str(query or "").lower())
        if not tokens:
            return rows

        def score(row: UserLongTermMemoryRecord) -> float:
            text = f"{row.memory_type} {row.value} {row.evidence}".lower()
            overlap = sum(1 for token in tokens if token.strip() and token in text)
            return float(overlap) + float(row.confidence or 0.0)

        return sorted(rows, key=score, reverse=True)

    def _clean_memory_candidates(self, memories: Any) -> list[dict[str, Any]]:
        if not isinstance(memories, list):
            return []
        clean = []
        for item in memories:
            if not isinstance(item, dict) or not item.get("value"):
                continue
            memory_type = str(item.get("type") or item.get("memory_type") or "other")
            candidate = {
                "type": memory_type,
                "memory_key": str(item.get("memory_key") or "").strip(),
                "value": str(item.get("value") or "").strip(),
                "confidence": self._float(item.get("confidence")),
                "scope": str(item.get("scope") or "long_term"),
                "evidence": str(item.get("evidence") or "").strip(),
            }
            candidate["memory_key"] = self._memory_key(candidate)
            clean.append(candidate)
        return clean

    def _memory_key(self, item: dict[str, Any]) -> str:
        raw = str(item.get("memory_key") or "").strip()
        if raw:
            return raw[:160]
        memory_type = str(item.get("type") or item.get("memory_type") or "other")
        value = str(item.get("value") or "").strip()
        normalized = value.replace(" ", "")[:80] or "general"
        return f"{memory_type}:{normalized}"[:160]

    def _memory_merge_key(self, memory_key: str) -> str:
        parts = [part for part in str(memory_key or "").split(":") if part]
        if len(parts) >= 2 and parts[0] == "other":
            return ":".join(parts[:2])[:160]
        return str(memory_key or "")[:160]

    def _row_memory_key(self, row: UserLongTermMemoryRecord) -> str:
        key = str(row.memory_key or "").strip()
        if key:
            return key
        return self._memory_key(
            {
                "type": row.memory_type,
                "value": row.value,
            }
        )

    def _row_to_memory_dict(self, row: UserLongTermMemoryRecord) -> dict[str, Any]:
        memory_key = self._row_memory_key(row)
        return {
            "id": row.id,
            "type": row.memory_type,
            "memory_key": memory_key,
            "merge_key": self._memory_merge_key(memory_key),
            "state": row.state,
            "value": row.value,
            "confidence": row.confidence,
            "evidence": row.evidence,
            "session_id": row.session_id,
            "observation_count": row.observation_count,
            "created_at": row.created_at.isoformat() if row.created_at else "",
            "last_observed_at": row.last_observed_at.isoformat()
            if row.last_observed_at
            else "",
        }

    def _clean_merged_memory(
        self,
        fallback: dict[str, Any],
        merged: Any,
    ) -> dict[str, Any]:
        if not isinstance(merged, dict):
            merged = {}
        item = {
            "type": str(
                merged.get("type")
                or merged.get("memory_type")
                or fallback.get("type")
                or "other"
            ),
            "memory_key": str(
                merged.get("memory_key") or fallback.get("memory_key") or ""
            ).strip(),
            "value": str(merged.get("value") or fallback.get("value") or "").strip(),
            "confidence": self._float(
                merged.get("confidence", fallback.get("confidence", 0.0))
            ),
            "scope": "long_term",
            "evidence": str(
                merged.get("evidence") or fallback.get("evidence") or ""
            ).strip(),
        }
        item["memory_key"] = self._memory_key(item)
        return item

    def _expires_at(
        self,
        memory_type: str,
        observed_at: datetime | None = None,
    ) -> datetime:
        days = LONG_MEMORY_TTL_DAYS.get(str(memory_type or "other"), LONG_MEMORY_TTL_DAYS["other"])
        return (observed_at or datetime.utcnow()) + timedelta(days=days)

    def _expire_active_long_term_memories(self, user_id: str) -> None:
        now = datetime.utcnow()
        with UserSessionLocal() as session:
            rows = (
                session.query(UserLongTermMemoryRecord)
                .filter(
                    UserLongTermMemoryRecord.user_id == user_id,
                    UserLongTermMemoryRecord.state == "active",
                    UserLongTermMemoryRecord.expires_at.is_not(None),
                    UserLongTermMemoryRecord.expires_at <= now,
                )
                .all()
            )
            if not rows:
                return
            for row in rows:
                row.state = "stale"
                row.updated_at = now
            session.commit()

    def _save(self, memory: SessionMemory) -> None:
        key = self.session_key(memory.user_id, memory.session_id)
        with UserSessionLocal() as session:
            record = session.get(ConversationSessionRecord, key)
            if not record:
                record = self._memory_to_record(memory)
                session.add(record)
            else:
                self._update_record(record, memory)
            session.commit()

    def _delete_session(self, session_key: str) -> None:
        with UserSessionLocal() as session:
            record = session.get(ConversationSessionRecord, session_key)
            if record:
                session.delete(record)
                session.commit()

    async def _apply_long_term_memories(
        self,
        memory: SessionMemory,
        memories: list[dict[str, Any]],
        raw_text: str = "",
        source: str = "session_end",
    ) -> list[dict[str, Any]]:
        saved: list[dict[str, Any]] = []
        for item in memories:
            if str(item.get("scope") or "long_term") == "session":
                self._save_memory_observation(
                    memory=memory,
                    item=item,
                    raw_text=raw_text,
                    action="ignore_new",
                    reason="session_scoped_memory",
                )
                continue
            result = await self._apply_one_long_term_memory(
                memory=memory,
                item=item,
                raw_text=raw_text,
                source=source,
            )
            if result:
                saved.append(result)
        return saved

    async def _apply_one_long_term_memory(
        self,
        memory: SessionMemory,
        item: dict[str, Any],
        raw_text: str,
        source: str,
    ) -> dict[str, Any]:
        memory_key = self._memory_key(item)
        item = {**item, "memory_key": memory_key}
        existing = self._find_existing_memory(memory.user_id, memory_key)
        if not existing:
            row_id = self._insert_long_term_memory(memory, item, source)
            self._save_memory_observation(
                memory=memory,
                item=item,
                raw_text=raw_text,
                action="insert",
                resolved_memory_id=row_id,
            )
            return {**item, "id": row_id, "state": "active", "action": "insert"}

        resolution = await self.memory_update_agent.resolve_memory(
            existing_memory=self._row_to_memory_dict(existing),
            new_memory=item,
            current_user_message=raw_text,
        )
        action = resolution["action"]
        reason = resolution["reason"]
        merged = self._clean_merged_memory(item, resolution.get("merged_memory"))

        if action == "merge":
            row_id = self._merge_long_term_memory(existing.id, merged)
            resolved = {**merged, "id": row_id, "state": "active", "action": "merge"}
        elif action == "replace":
            row_id = self._replace_long_term_memory(memory, existing.id, merged, source)
            resolved = {**merged, "id": row_id, "state": "active", "action": "replace"}
        elif action == "reject_old":
            self._reject_long_term_memory(existing.id)
            row_id = None
            resolved = {}
        elif action == "ignore_new":
            row_id = None
            resolved = {}
        else:
            row_id = self._insert_long_term_memory(memory, item, source)
            resolved = {**item, "id": row_id, "state": "active", "action": "keep_both"}

        self._save_memory_observation(
            memory=memory,
            item=item,
            raw_text=raw_text,
            action=action,
            existing_memory_id=existing.id,
            resolved_memory_id=row_id,
            reason=reason,
        )
        return resolved

    def _find_existing_memory(
        self,
        user_id: str,
        memory_key: str,
    ) -> UserLongTermMemoryRecord | None:
        merge_key = self._memory_merge_key(memory_key)
        with UserSessionLocal() as session:
            rows = (
                session.query(UserLongTermMemoryRecord)
                .filter(
                    UserLongTermMemoryRecord.user_id == user_id,
                    UserLongTermMemoryRecord.state.in_(["active", "stale"]),
                )
                .order_by(UserLongTermMemoryRecord.updated_at.desc())
                .limit(100)
                .all()
            )
            for row in rows:
                row_key = self._row_memory_key(row)
                if row_key == memory_key or self._memory_merge_key(row_key) == merge_key:
                    return row
        return None

    def _insert_long_term_memory(
        self,
        memory: SessionMemory,
        item: dict[str, Any],
        source: str,
    ) -> int:
        now = datetime.utcnow()
        memory_type = str(item.get("type") or "other")
        with UserSessionLocal() as session:
            row = UserLongTermMemoryRecord(
                user_id=memory.user_id,
                session_id=memory.session_id,
                memory_type=memory_type,
                memory_key=self._memory_key(item),
                state="active",
                value=str(item.get("value") or ""),
                confidence=self._float(item.get("confidence")),
                evidence=str(item.get("evidence") or ""),
                source=source,
                observation_count=1,
                created_at=now,
                updated_at=now,
                last_observed_at=now,
                expires_at=self._expires_at(memory_type, now),
            )
            session.add(row)
            session.commit()
            return int(row.id)

    def _merge_long_term_memory(
        self,
        row_id: int,
        item: dict[str, Any],
    ) -> int:
        now = datetime.utcnow()
        with UserSessionLocal() as session:
            row = session.get(UserLongTermMemoryRecord, row_id)
            if not row:
                return row_id
            row.memory_type = str(item.get("type") or row.memory_type or "other")
            row.memory_key = self._memory_key(item)
            row.value = str(item.get("value") or row.value or "")
            row.confidence = max(
                float(row.confidence or 0.0),
                self._float(item.get("confidence")),
            )
            row.evidence = str(item.get("evidence") or row.evidence or "")
            row.state = "active"
            row.observation_count = int(row.observation_count or 0) + 1
            row.updated_at = now
            row.last_observed_at = now
            row.expires_at = self._expires_at(row.memory_type, now)
            session.commit()
            return row_id

    def _replace_long_term_memory(
        self,
        memory: SessionMemory,
        old_row_id: int,
        item: dict[str, Any],
        source: str,
    ) -> int:
        now = datetime.utcnow()
        memory_type = str(item.get("type") or "other")
        with UserSessionLocal() as session:
            old = session.get(UserLongTermMemoryRecord, old_row_id)
            new = UserLongTermMemoryRecord(
                user_id=memory.user_id,
                session_id=memory.session_id,
                memory_type=memory_type,
                memory_key=self._memory_key(item),
                state="active",
                value=str(item.get("value") or ""),
                confidence=self._float(item.get("confidence")),
                evidence=str(item.get("evidence") or ""),
                source=source,
                observation_count=1,
                created_at=now,
                updated_at=now,
                last_observed_at=now,
                expires_at=self._expires_at(memory_type, now),
            )
            session.add(new)
            session.flush()
            if old:
                old.state = "superseded"
                old.superseded_by_id = new.id
                old.updated_at = now
            session.commit()
            return int(new.id)

    def _reject_long_term_memory(self, row_id: int) -> None:
        with UserSessionLocal() as session:
            row = session.get(UserLongTermMemoryRecord, row_id)
            if row:
                row.state = "rejected"
                row.confidence = 0.0
                row.updated_at = datetime.utcnow()
                session.commit()

    def _save_memory_observation(
        self,
        memory: SessionMemory,
        item: dict[str, Any],
        raw_text: str,
        action: str,
        existing_memory_id: int | None = None,
        resolved_memory_id: int | None = None,
        reason: str = "",
    ) -> None:
        with UserSessionLocal() as session:
            session.add(
                MemoryObservationRecord(
                    user_id=memory.user_id,
                    session_id=memory.session_id,
                    memory_key=self._memory_key(item),
                    memory_type=str(item.get("type") or "other"),
                    value=str(item.get("value") or ""),
                    confidence=self._float(item.get("confidence")),
                    evidence=str(item.get("evidence") or ""),
                    scope=str(item.get("scope") or "long_term"),
                    raw_text=raw_text,
                    existing_memory_id=existing_memory_id,
                    resolved_memory_id=resolved_memory_id,
                    resolved_action=action,
                    resolution_reason=reason,
                )
            )
            session.commit()

    def _save_long_term_candidates(
        self,
        memory: SessionMemory,
        memories: list[dict[str, Any]],
    ) -> None:
        if not memories:
            return
        with UserSessionLocal() as session:
            existing = {
                (row.memory_type, row.value)
                for row in session.query(UserLongTermMemoryRecord)
                .filter(UserLongTermMemoryRecord.user_id == memory.user_id)
                .all()
            }
            for item in memories:
                memory_type = str(item.get("type") or "other")
                value = str(item.get("value") or "")
                if not value or (memory_type, value) in existing:
                    continue
                session.add(
                    UserLongTermMemoryRecord(
                        user_id=memory.user_id,
                        session_id=memory.session_id,
                        memory_type=memory_type,
                        memory_key=self._memory_key(item),
                        state="active",
                        value=value,
                        confidence=self._float(item.get("confidence")),
                        evidence=str(item.get("evidence") or ""),
                        last_observed_at=datetime.utcnow(),
                        expires_at=self._expires_at(memory_type),
                    )
                )
                existing.add((memory_type, value))
            session.commit()

    def _memory_to_record(self, memory: SessionMemory) -> ConversationSessionRecord:
        record = ConversationSessionRecord(
            session_key=self.session_key(memory.user_id, memory.session_id),
            user_id=memory.user_id,
            session_id=memory.session_id,
        )
        self._update_record(record, memory)
        return record

    def _update_record(
        self,
        record: ConversationSessionRecord,
        memory: SessionMemory,
    ) -> None:
        record.user_id = memory.user_id
        record.session_id = memory.session_id
        record.turn_count = memory.turn_count
        record.summary = memory.summary
        record.last_query = ""
        record.last_image_query = ""
        record.last_image_summary = ""
        record.last_image_attributes_json = "[]"
        record.active_constraints_json = "[]"
        record.last_categories_json = "[]"
        record.last_keywords_json = "[]"
        record.last_recommended_product_ids_json = "[]"
        record.messages_json = self._json(memory.messages)
        record.turns_json = self._json(memory.turns)
        record.long_term_candidates_json = self._json(memory.long_term_candidates)
        record.updated_at = datetime.utcnow()

    def _record_to_memory(self, record: ConversationSessionRecord) -> SessionMemory:
        return SessionMemory(
            user_id=record.user_id,
            session_id=record.session_id,
            created_at=record.created_at.timestamp() if record.created_at else time.time(),
            updated_at=record.updated_at.timestamp() if record.updated_at else time.time(),
            turn_count=record.turn_count,
            summary=record.summary or "",
            messages=self._dict_list(self._loads(record.messages_json, [])),
            turns=self._dict_list(self._loads(record.turns_json, [])),
            long_term_candidates=self._dict_list(
                self._loads(record.long_term_candidates_json, [])
            ),
        )

    def _tool_events(self, response: RecommendationResponse) -> list[dict[str, Any]]:
        events = []
        for name, result in response.agent_results.items():
            data = result.data or {}
            event: dict[str, Any] = {
                "agent": name,
                "success": result.success,
                "confidence": result.confidence,
            }
            if name == "product_rec":
                event["recall_strategy"] = getattr(result, "recall_strategy", "")
                event["recall_reason"] = data.get("recall_reason", "")
                event["query_recall_count"] = data.get("query_recall_count", 0)
                event["profile_recall_count"] = data.get("profile_recall_count", 0)
                event["selected_categories"] = data.get("selected_categories", [])
                event["selected_keywords"] = data.get("selected_keywords", [])
            elif name in {"conversation_understanding", "conversation_context"}:
                event["standalone_query"] = data.get("standalone_query", "")
                event["is_follow_up"] = data.get("is_follow_up", False)
                event["constraints"] = data.get("constraints", [])
            elif name in {"response_generation", "shopping_guide"}:
                event["answer"] = (getattr(result, "answer", "") or data.get("answer", ""))[:500]
            events.append(event)
        return events

    def _compact_context(self, context: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "query",
            "original_query",
            "image_query",
            "image_summary",
            "image_attributes",
            "categories",
            "category",
            "budget",
            "constraints",
            "avoid_categories",
        ]
        return {key: context[key] for key in keys if key in context and context[key]}

    def _should_summarize(self, memory: SessionMemory) -> bool:
        if self.summarize_every > 0 and memory.turn_count % self.summarize_every == 0:
            return True
        return self._pending_turn_tokens(memory) >= self.max_memory_tokens > 0

    def _pending_turn_tokens(self, memory: SessionMemory) -> int:
        payload = {
            "previous_summary": memory.summary,
            "recent_turns": memory.turns,
        }
        return estimate_tokens(json.dumps(payload, ensure_ascii=False))

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if item]
        if isinstance(value, str) and value:
            return [value]
        return []

    def _dict_list(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def _loads(self, raw: str | None, fallback: Any) -> Any:
        try:
            return json.loads(raw or "")
        except (TypeError, json.JSONDecodeError):
            return fallback

    def _json(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
