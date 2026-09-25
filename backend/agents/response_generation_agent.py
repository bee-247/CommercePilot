from __future__ import annotations

import json
import re
from typing import Any

from core.agent_config import get_agent_system_config
from core.model_clients import create_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from models.schemas import AgentResult, ChatHistoryMessage, Product, ShoppingGuideResult
from pydantic import BaseModel
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
10. 字段出现在 unverified_requirements 时，必须保守表达并提醒用户确认，不能将其写成确定能力。
13. review_feedback 是上一版最终回复的审核问题，修订时必须逐项解决，不能只修改评分。
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

CONVERSATION_PROMPT = """你是电商导购聊天Agent。你只输出JSON，不要解释。

当前用户消息不需要重新召回商品。请基于最近对话和会话记忆，给出自然、简短、有帮助的回复。

要求:
1. 对问候、感谢、确认，正常回应，不要硬推荐商品。
2. 对上一轮商品追问，可以基于 recent_history 和 session_memory 解释；如果缺少商品细节，要坦诚说明，并引导用户指出商品名称或序号。
3. 对偏好更新，简短确认已经理解，不要声称已经永久写入系统。
4. 商品事实以product_facts为准，历史回复不能作为商品事实证据；不要编造商品参数、价格、库存或优惠。
6. 有review_feedback时必须按审核意见修订回复。
5. 回复应像导购助理，不要输出Markdown表格。

输出格式:
{
  "answer": "给用户的回复"
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


class ResponseGenerationAgent(BaseAgent):
    def __init__(self):
        agent_config = get_agent_system_config()
        definition = agent_config.agent("response-generation")
        model_config = agent_config.resolved_model("response-generation")
        super().__init__(
            name="response_generation",
            timeout=definition.runtime.timeout_seconds,
            max_retries=definition.runtime.max_attempts,
        )
        self.llm = create_chat_model(
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            enable_thinking=False,
            timeout=definition.runtime.timeout_seconds,
            max_retries=0,
            stream_usage=True,
        )
        self.conversation_model = self.llm.bind(
            **agent_config.model_profiles["conversational"].model_dump()
        )

    async def _execute(self, **kwargs: Any) -> AgentResult:
        mode = kwargs.get("mode", "recommendation")
        if mode == "conversation":
            return await self._reply_to_conversation(**kwargs)
        if mode != "recommendation":
            raise ValueError(f"未知回复模式: {mode}")
        return await self._reply_to_recommendation(**kwargs)

    async def _reply_to_recommendation(self, **kwargs: Any) -> ShoppingGuideResult:
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
                product
                for product in products
                if product.category in requested_categories
            ]
            # 类目只用于收窄展示，不作为拦截条件：过滤为空时保留全部召回结果，
            # 由语义召回的相关性兜底，避免误伤正常推荐。
            if matched_products:
                products = matched_products

        if requested_keywords:
            matched_products = self._products_matching_keywords(
                products, requested_keywords
            )
            # 关键词同理：无命中时降级为不过滤，保证导购回复始终有商品可讲。
            if matched_products:
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
            "shopping_goals": shopping_goals
            if isinstance(shopping_goals, list)
            else [],
            "candidate_evaluations": [
                item
                for item in (context.get("candidate_evaluations") or [])
                if isinstance(item, dict)
                and str(item.get("product_id") or "") in displayed_product_ids
            ],
            "review_feedback": kwargs.get("feedback") or [],
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
        decision_summary = self._clean_decision_summary(data.get("decision_summary"))
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
            confidence=0.85,
        )

    def _clean_decision_summary(self, raw: Any) -> dict[str, list[str]]:
        if not isinstance(raw, dict):
            return {
                "requirements": [],
                "selection_reasons": [],
                "uncertainties": [],
            }
        return {
            key: [str(item).strip() for item in raw.get(key, []) if str(item).strip()]
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
        product_categories = sorted(
            {product.category for product in products if product.category}
        )
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

    async def _reply_to_conversation(self, **kwargs: Any) -> AgentResult:
        message = str(kwargs.get("message") or "").strip()
        history: list[ChatHistoryMessage] = kwargs.get("history", [])
        context: dict[str, Any] = kwargs.get("context", {})

        if self._is_simple_ack(message):
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={
                    "answer": self._simple_ack_answer(message),
                    "deterministic_reply": True,
                },
                confidence=0.95,
            )

        payload = {
            "current_message": message,
            "product_facts": context.get("product_facts") or [],
            "review_feedback": kwargs.get("feedback") or [],
            "route": context.get("chat_route", {}),
            "session_memory": context.get("session_memory", {}),
            "recent_history": [
                {"role": item.role, "content": item.content}
                for item in history[-8:]
                if item.content
            ],
        }
        try:
            response = await self.conversation_model.ainvoke(
                [
                    SystemMessage(content=CONVERSATION_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            )
            data = parse_json_object(response.content)
            answer = str(data.get("answer") or "").strip()
        except Exception:
            answer = ""

        if not answer:
            answer = (
                "好的，我明白了。你如果想继续看商品或对比刚才的推荐，可以直接告诉我。"
            )
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

    def generate_structured(
        self,
        task_name: str,
        payload: dict,
        schema: type[BaseModel],
        feedback: list[dict] | None = None,
        previous_content: dict | None = None,
    ) -> dict:
        parameters = get_agent_system_config().model_profiles["structured_response"]
        prompt = (
            f"你负责{task_name}。严格依据context_chunks原文和需求输出结构化结果。"
            "不得编造商品参数、库存、优惠、物流或售后承诺；证据不足写入limitation。"
            "source_chunk_ids只能来自context_chunks。若有反馈，请修订previous_content。\n"
            + json.dumps(
                {
                    "payload": payload,
                    "feedback": feedback or [],
                    "previous_content": previous_content,
                },
                ensure_ascii=False,
            )
        )
        result = self.llm.with_structured_output(schema).invoke(
            prompt, temperature=parameters.temperature, max_tokens=parameters.max_tokens
        )
        return schema.model_validate(result).model_dump()

    def build_customer_service_graph(self):
        from customer_service.agents.executor import build_customer_service_agent

        return build_customer_service_agent(self.llm)
