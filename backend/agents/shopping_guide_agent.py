from __future__ import annotations

import json
import re
from typing import Any

from core.config import get_settings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from models.schemas import Product, ShoppingGuideResult
from utils.json_utils import parse_json_object

from .base_agent import BaseAgent

SYSTEM_PROMPT = """你是电商智能导购Agent。你需要基于用户需求、会话记忆、长期记忆、推荐商品和可选知识库证据，给出自然、可信、可执行的导购建议，并为每个商品生成短文案。

要求:
1. 先用一句话说明你理解的需求。
2. 给出最推荐商品和推荐理由。
3. 如有多个商品，做简短对比，说明适合人群或场景。
4. 提醒用户需要确认的关键点，例如预算、兼容性、规格、售后、库存。
5. 只能依据提供的商品信息回答，不要编造不存在的参数。
6. 为每个商品生成一条30字以内的卖点文案，避免“最好、第一、绝对、100%”等夸张广告词。
7. 知识库证据只能用于解释通用原理、选购方法和注意事项，不能覆盖或虚构商品参数。
8. 使用知识库证据时，在回答中以“资料《文件名》第X页”标明来源；没有证据时不要编造引用。
9. 优先依据候选评估中已通过硬约束、匹配分更高且证据充分的商品组织回答。
10. 校验未通过或字段出现在 unverified_requirements 时，必须保守表达并提醒用户确认，不能将其写成确定能力。
11. decision_summary 只给出可验证的决策摘要，不要输出内部思维过程。
12. 只输出JSON对象，不要解释。

输出格式:
{
  "answer": "导购回答纯文本，不要使用Markdown格式，不要使用#、**、表格竖线、引用符号>",
  "product_copies": [
    {"product_id": "xxx", "copy": "商品短文案"}
  ],
  "decision_summary": {
    "requirements": ["本轮关键需求"],
    "selection_reasons": ["有商品字段支持的选择理由"],
    "uncertainties": ["仍需确认的信息"]
  }
}
"""

FORBIDDEN_WORDS = [
    "最好",
    "第一",
    "国家级",
    "全球首",
    "绝对",
    "100%",
    "永久",
    "万能",
    "祖传",
    "纯天然",
]


class ShoppingGuideAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(name="shopping_guide", timeout=12.0)
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.4,
            max_tokens=1200,
        )

    async def _execute(self, **kwargs: Any) -> ShoppingGuideResult:
        query: str = kwargs.get("query", "")
        context: dict[str, Any] = kwargs.get("context", {})
        products: list[Product] = kwargs.get("products", [])
        requested_categories: list[str] = kwargs.get("requested_categories", [])
        requested_keywords: list[str] = kwargs.get("requested_keywords", [])
        image_summary: str = kwargs.get("image_summary", "")
        shopping_goals = context.get("shopping_goals", [])

        if not products:
            had_evaluations = bool(context.get("candidate_evaluations"))
            answer = (
                "当前候选商品均未通过本轮需求约束，建议调整预算、偏好或补充关键条件后再试。"
                if had_evaluations
                else "我已经理解你的需求，但当前商品库或库存里没有可推荐的商品。"
            )
            return ShoppingGuideResult(
                success=True,
                answer=answer,
                data={
                    "displayed_product_ids": [],
                    "decision_summary": {
                        "requirements": [],
                        "selection_reasons": [],
                        "uncertainties": [answer],
                    },
                },
                confidence=1.0,
            )

        requested_categories = self._requested_categories(
            query=query,
            products=products,
            requested_categories=requested_categories,
        )
        if requested_categories:
            matched_products = [
                product for product in products if product.category in requested_categories
            ]
            if not matched_products:
                category_text = "、".join(requested_categories)
                return ShoppingGuideResult(
                    success=True,
                    answer=(
                        f"我理解你想买{category_text}，但当前召回结果里没有匹配的"
                        f"{category_text}商品。建议先补充或重新生成相关类目商品数据，"
                        "再进行导购推荐。"
                    ),
                    data={
                        "requested_categories": requested_categories,
                        "displayed_product_ids": [],
                    },
                    confidence=0.8,
                )
            products = matched_products

        if requested_keywords:
            matched_products = self._products_matching_keywords(products, requested_keywords)
            if not matched_products:
                keyword_text = "、".join(requested_keywords)
                return ShoppingGuideResult(
                    success=True,
                    answer=(
                        f"我理解你想买{keyword_text}，但当前召回结果里没有匹配的商品。"
                        "建议先补充或重新生成对应商品数据，再进行导购推荐。"
                    ),
                    data={
                        "requested_keywords": requested_keywords,
                        "displayed_product_ids": [],
                    },
                    confidence=0.8,
                )
            products = matched_products

        displayed_product_ids = {product.product_id for product in products}
        payload = {
            "user_query": query,
            "image_summary": image_summary,
            "session_memory": context.get("session_memory", {}),
            "long_term_memory": context.get("long_term_memory", []),
            "conversation_context": context.get("conversation_context", {}),
            "copy_style": (
                context.get("experiment_config", {}).get("style", "natural")
                if isinstance(context.get("experiment_config"), dict)
                else "natural"
            ),
            "shopping_goals": shopping_goals if isinstance(shopping_goals, list) else [],
            "candidate_evaluations": [
                item
                for item in (context.get("candidate_evaluations") or [])
                if isinstance(item, dict)
                and str(item.get("product_id") or "") in displayed_product_ids
            ],
            "recommendation_audit": context.get("recommendation_audit") or {},
            "rag_evidence": [
                {
                    "filename": item.get("filename", ""),
                    "page_number": item.get("page_number"),
                    "section_title": item.get("section_title", ""),
                    "text": str(item.get("text") or "")[:800],
                }
                for item in (context.get("rag_context") or [])[:6]
                if isinstance(item, dict)
            ],
            "products": [
                {
                    "product_id": product.product_id,
                    "name": product.name,
                    "category": product.category,
                    "price": product.price,
                    "brand": product.brand,
                    "stock": product.stock,
                    "tags": product.tags,
                    "description": product.description,
                }
                for product in products
            ],
        }

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        )
        data = parse_json_object(response.content)
        answer = str(data.get("answer") or response.content)
        copies = self._clean_copies(data.get("product_copies"), products)
        decision_summary = self._clean_decision_summary(
            data.get("decision_summary")
        )
        audit = context.get("recommendation_audit") or {}
        audit_passed = bool(
            isinstance(audit, dict) and audit.get("passed", False)
        )

        return ShoppingGuideResult(
            success=True,
            answer=self._clean_answer(answer),
            copies=copies,
            data={
                "product_copies": copies,
                "displayed_product_ids": [product.product_id for product in products],
                "requested_categories": requested_categories,
                "decision_summary": decision_summary,
            },
            confidence=0.9 if audit_passed else 0.65,
        )

    def _clean_decision_summary(self, raw: Any) -> dict[str, list[str]]:
        if not isinstance(raw, dict):
            return {
                "requirements": [],
                "selection_reasons": [],
                "uncertainties": [],
            }
        return {
            key: [
                str(item).strip()
                for item in raw.get(key, [])
                if str(item).strip()
            ]
            if isinstance(raw.get(key), list)
            else []
            for key in (
                "requirements",
                "selection_reasons",
                "uncertainties",
            )
        }

    def _clean_answer(self, raw: str) -> str:
        text = str(raw or "").strip()
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = text.replace("**", "")
        text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n\s*\|[-|:\s]+\|\s*\n", "\n", text)
        text = re.sub(r"^\s*\|", "", text, flags=re.MULTILINE)
        text = re.sub(r"\|\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _clean_copies(
        self,
        raw: Any,
        products: list[Product],
    ) -> list[dict[str, str]]:
        product_ids = {product.product_id for product in products}
        if not isinstance(raw, list):
            return []
        copies = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            product_id = str(item.get("product_id") or "")
            copy = str(item.get("copy") or "").strip()
            if not product_id or product_id not in product_ids or not copy:
                continue
            for word in FORBIDDEN_WORDS:
                copy = re.sub(re.escape(word), "***", copy)
            copies.append({"product_id": product_id, "copy": copy})
        return copies

    def _requested_categories(
        self,
        query: str,
        products: list[Product],
        requested_categories: list[str],
    ) -> list[str]:
        categories = []
        for category in requested_categories:
            category = str(category)
            if category and category not in categories:
                categories.append(category)
        product_categories = sorted({product.category for product in products if product.category})
        for category in product_categories:
            if category and category in query and category not in categories:
                categories.append(category)
        return categories

    def _products_matching_keywords(
        self,
        products: list[Product],
        keywords: list[str],
    ) -> list[Product]:
        matched = []
        for product in products:
            text = " ".join(
                [
                    product.name,
                    product.category,
                    " ".join(product.tags),
                ]
            )
            if any(keyword and keyword in text for keyword in keywords):
                matched.append(product)
        return matched
