from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.config import get_settings
from models.schemas import AgentResult
from utils.json_utils import parse_json_object

from .base_agent import BaseAgent


EXTRACT_PROMPT = """你是电商长期记忆抽取Agent。你只输出JSON，不要解释。

请从用户表达和会话摘要中抽取值得写入长期记忆的稳定偏好。不要抽取一次性需求，例如“今天想买”“这次送人”“这张图类似款”，除非用户明确说这是长期偏好。

每条记忆必须包含 memory_key。memory_key 用于表示同一个偏好槽位，格式建议:
- preferred_category:类目或对象
- style_tag:对象或场景
- price_preference:对象或类目
- disliked_category:类目或对象
- usage_scene:对象或类目
- other:主题，例如 other:brand_trust、other:service_preference、other:operation_preference

如果 type=other，memory_key 应尽量使用“other:主题”两段式，不要因为商品类目过度拆分成很多 key。具体适用范围可以写在 value 里，例如“选家电时偏好售后稳定”，memory_key 仍可用 other:service_preference。

scope:
- long_term: 稳定偏好，可以写入长期记忆
- session: 当前轮/本次需求，不应写入长期记忆

输出格式:
{
  "memories": [
    {
      "type": "preferred_category" | "style_tag" | "price_preference" | "disliked_category" | "usage_scene" | "other",
      "memory_key": "price_preference:earphones",
      "value": "记忆内容",
      "confidence": 0.0,
      "scope": "long_term" | "session",
      "evidence": "来自用户表达或会话的简短证据"
    }
  ],
  "reason": "简短原因"
}
"""


RESOLVE_PROMPT = """你是电商长期记忆合并Agent。你只输出JSON，不要解释。

系统发现 existing_memory 和 new_memory 属于相同或相近的记忆槽位。请判断如何处理。

可选 action:
- merge: 新旧语义一致或互补，合并成一条更准确的记忆。
- replace: 用户明确表达长期偏好变化，新记忆应替代旧记忆。
- keep_both: 同槽位但作用域或场景确实不同，且合并会损失重要差异，可以并存。
- ignore_new: 新记忆只是临时需求、证据太弱或不适合写长期记忆。
- reject_old: 用户明确否认旧记忆或要求不要记。

要求:
1. 如果 new_memory.scope 是 session，通常选择 ignore_new 或 keep_both，不要替代长期记忆。
2. 只有用户表达“以后、一直、通常、我现在偏好变了、不要再按旧偏好”等长期变化时，才选择 replace。
3. merge 时输出 merged_memory，value 要简短，不要堆砌证据。
4. replace 时输出 merged_memory，表示新 active 记忆。
5. 如果 type=other，尽量保持 memory_key 为 other:主题 两段式，并把类目/对象范围写进 value。
6. 不要编造用户没有表达过的信息。

输出格式:
{
  "action": "merge" | "replace" | "keep_both" | "ignore_new" | "reject_old",
  "merged_memory": {
    "type": "style_tag",
    "memory_key": "style_tag:earphones",
    "value": "合并或替换后的记忆内容",
    "confidence": 0.0,
    "evidence": "合并后的简短证据"
  },
  "reason": "简短原因"
}
"""


class MemoryUpdateAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(name="memory_update", timeout=8.0)
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.0,
            max_tokens=900,
        )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        data = await self.extract_memories(**kwargs)
        return AgentResult(
            agent_name=self.name,
            success=True,
            data=data,
            confidence=0.85,
        )

    async def extract_memories(self, **kwargs: Any) -> dict[str, Any]:
        payload = {
            "user_message": kwargs.get("message", ""),
            "session_memory": kwargs.get("session_memory", {}),
            "messages": kwargs.get("messages", []),
        }
        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=EXTRACT_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            )
            data = parse_json_object(response.content)
        except Exception as exc:
            data = {"memories": [], "reason": f"memory_extract_failed: {exc}"}
        memories = data.get("memories")
        if not isinstance(memories, list):
            memories = []
        return {
            "memories": [
                item for item in memories if isinstance(item, dict) and item.get("value")
            ],
            "reason": str(data.get("reason") or ""),
        }

    async def resolve_memory(
        self,
        existing_memory: dict[str, Any],
        new_memory: dict[str, Any],
        current_user_message: str = "",
    ) -> dict[str, Any]:
        payload = {
            "existing_memory": existing_memory,
            "new_memory": new_memory,
            "current_user_message": current_user_message,
        }
        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=RESOLVE_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            )
            data = parse_json_object(response.content)
        except Exception as exc:
            data = {
                "action": "keep_both",
                "merged_memory": None,
                "reason": f"memory_resolve_failed: {exc}",
            }
        action = str(data.get("action") or "keep_both")
        if action not in {"merge", "replace", "keep_both", "ignore_new", "reject_old"}:
            action = "keep_both"
        merged = data.get("merged_memory")
        if not isinstance(merged, dict):
            merged = {}
        return {
            "action": action,
            "merged_memory": merged,
            "reason": str(data.get("reason") or ""),
        }
