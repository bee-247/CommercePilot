from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.config import get_settings
from models.schemas import AgentResult
from utils.json_utils import parse_json_object

from .base_agent import BaseAgent


SYSTEM_PROMPT = """你是电商图片理解Agent。根据用户上传的图片、用户文字需求和当前商品库可用类目，提取适合商品推荐召回的信息。

请只输出JSON:
{
  "query": "适合用于向量召回的中文搜索描述",
  "category": "必须从可用类目中选择的最适合类目；如果没有合适类目则为空字符串",
  "attributes": ["颜色", "材质", "风格", "用途等关键词"],
  "summary": "一句话说明你从图片中看到了什么"
}

规则:
1. category 只能从用户提供的可用类目列表中选择，不能自己创造类目。
2. 如果图片中物品和可用类目没有明显匹配，category 返回空字符串。
3. query 可以保留图片真实识别结果和用户需求，用于语义召回。
4. 如果图片不是商品，也要根据画面里的场景和用户文字推断可能需要的商品。"""


class ImageUnderstandingAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(name="image_understanding", timeout=10.0)
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.2,
            max_tokens=800,
        )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        image_data_url: str = kwargs["image_data_url"]
        message: str = kwargs.get("message", "")
        available_categories: list[str] = kwargs.get("available_categories", [])
        category_text = "、".join(available_categories) if available_categories else "无"

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                f"用户文字需求: {message or '无'}\n"
                                f"当前商品库可用类目: {category_text}\n"
                                "请从可用类目中选择最适合的 category。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ]
                ),
            ]
        )
        data = parse_json_object(response.content)
        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "raw_response": response.content,
                "query": data.get("query", message),
                "category": data.get("category", ""),
                "attributes": data.get("attributes", []),
                "summary": data.get("summary", ""),
            },
            confidence=0.85,
        )
