from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.config import get_settings
from models.schemas import AgentResult, ChatHistoryMessage
from utils.json_utils import parse_json_object

from .base_agent import BaseAgent


SYSTEM_PROMPT = """你是电商导购聊天Agent。你只输出JSON，不要解释。

当前用户消息不需要重新召回商品。请基于最近对话和会话记忆，给出自然、简短、有帮助的回复。

要求:
1. 对问候、感谢、确认，正常回应，不要硬推荐商品。
2. 对上一轮商品追问，可以基于 recent_history 和 session_memory 解释；如果缺少商品细节，要坦诚说明，并引导用户指出商品名称或序号。
3. 对偏好更新，简短确认已经理解，不要声称已经永久写入系统。
4. 不要编造商品参数、价格、库存或优惠。
5. 回复应像导购助理，不要输出Markdown表格。

输出格式:
{
  "answer": "给用户的回复"
}
"""


class ConversationReplyAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(name="conversation_reply", timeout=5.0)
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=500,
        )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        message = str(kwargs.get("message") or "").strip()
        history: list[ChatHistoryMessage] = kwargs.get("history", [])
        context: dict[str, Any] = kwargs.get("context", {})

        if self._is_simple_ack(message):
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={"answer": self._simple_ack_answer(message)},
                confidence=0.95,
            )

        payload = {
            "current_message": message,
            "route": context.get("chat_route", {}),
            "session_memory": context.get("session_memory", {}),
            "recent_history": [
                {"role": item.role, "content": item.content}
                for item in history[-8:]
                if item.content
            ],
        }
        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            )
            data = parse_json_object(response.content)
            answer = str(data.get("answer") or "").strip()
        except Exception:
            answer = ""

        if not answer:
            answer = "好的，我明白了。你如果想继续看商品或对比刚才的推荐，可以直接告诉我。"
        return AgentResult(
            agent_name=self.name,
            success=True,
            data={"answer": answer},
            confidence=0.85,
        )

    def _is_simple_ack(self, message: str) -> bool:
        compact = message.strip().lower().replace(" ", "")
        return compact in {
            "谢谢",
            "谢谢你",
            "感谢",
            "好的",
            "好",
            "ok",
            "okay",
            "了解",
            "明白",
            "知道了",
            "可以",
            "再见",
            "拜拜",
            "你好",
            "您好",
        }

    def _simple_ack_answer(self, message: str) -> str:
        compact = message.strip().lower().replace(" ", "")
        if compact in {"你好", "您好"}:
            return "你好，我在。想随便聊聊商品，还是继续看刚才的推荐都可以。"
        if compact in {"再见", "拜拜"}:
            return "好的，之后需要继续挑商品时再来找我。"
        if compact in {"谢谢", "谢谢你", "感谢"}:
            return "不客气，能帮你挑到合适的就好。"
        return "好的，我明白了。"
