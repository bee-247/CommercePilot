from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.config import get_settings
from models.schemas import AgentResult, ChatHistoryMessage
from utils.json_utils import parse_json_object

from .base_agent import BaseAgent


SYSTEM_PROMPT = """你是电商导购意图识别Agent。你只输出JSON，不要解释。

请判断当前用户消息包含哪些意图。注意：同一句话可能同时需要商品推荐和长期记忆更新。

路由类型:
- recommendation: 用户明确想买、想推荐、想找商品、换一批商品、提出新的购买需求。
- product_follow_up: 用户追问上一轮推荐商品的原因、区别、价格、适用场景、对比、某个序号。
- preference_update: 用户表达长期偏好、排斥项、预算习惯、风格变化、要求记住或不要记住某偏好。
- recommendation_with_preference_update: 用户同时表达长期偏好变化并要求推荐商品。
- smalltalk: 问候、感谢、确认、结束语、闲聊。
- unsupported: 与购物导购无关且不适合回答的任务。

规则:
1. needs_recommendation 表示是否要进入商品召回和推荐链路。
2. needs_memory_update 表示是否要尝试抽取并更新长期记忆。
3. 两个字段彼此独立，可以同时为 true。
4. “这次/今天/送人/临时”的一次性需求通常不需要更新长期记忆，但仍可能需要推荐。
5. “以后/一直/通常/我现在更喜欢/不要再按这个推荐/别记这个”等长期偏好或记忆管理表达，通常需要更新长期记忆。
6. 如果用户同时说“以后耳机预算2000左右，帮我推荐一款”，needs_memory_update 和 needs_recommendation 都应为 true。

输出格式:
{
  "route": "recommendation" | "product_follow_up" | "preference_update" | "recommendation_with_preference_update" | "smalltalk" | "unsupported",
  "needs_recommendation": true,
  "needs_memory_update": false,
  "memory_update_reason": "为什么需要或不需要更新长期记忆",
  "reason": "简短原因"
}
"""


class ChatRouterAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(name="chat_router", timeout=4.0)
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.0,
            max_tokens=300,
        )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        message = str(kwargs.get("message") or "").strip()
        history: list[ChatHistoryMessage] = kwargs.get("history", [])

        if not message:
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={
                    "route": "smalltalk",
                    "needs_recommendation": False,
                    "needs_memory_update": False,
                    "memory_update_reason": "空消息不需要更新长期记忆",
                    "reason": "空消息不需要处理",
                },
                confidence=0.95,
            )

        payload = {
            "current_message": message,
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
        except Exception as exc:
            data = self._fallback_route(message, str(exc))

        route = str(data.get("route") or "recommendation")
        if route not in {
            "recommendation",
            "product_follow_up",
            "preference_update",
            "recommendation_with_preference_update",
            "smalltalk",
            "unsupported",
        }:
            route = ""
        needs_recommendation = bool(data.get("needs_recommendation", False))
        needs_memory_update = bool(data.get("needs_memory_update", False))
        if not route:
            route = self._route_from_flags(needs_recommendation, needs_memory_update)
        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "route": route,
                "needs_recommendation": needs_recommendation,
                "needs_memory_update": needs_memory_update,
                "memory_update_reason": str(data.get("memory_update_reason") or ""),
                "reason": str(data.get("reason") or ""),
            },
            confidence=0.85,
        )

    def _fallback_route(self, message: str, error: str) -> dict[str, Any]:
        return {
            "route": "recommendation",
            "needs_recommendation": True,
            "needs_memory_update": False,
            "memory_update_reason": "意图识别失败时不更新长期记忆",
            "reason": f"router_fallback: {error}",
        }

    def _route_from_flags(
        self,
        needs_recommendation: bool,
        needs_memory_update: bool,
    ) -> str:
        if needs_recommendation and needs_memory_update:
            return "recommendation_with_preference_update"
        if needs_recommendation:
            return "recommendation"
        if needs_memory_update:
            return "preference_update"
        return "smalltalk"
