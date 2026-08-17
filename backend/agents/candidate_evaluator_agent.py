from __future__ import annotations

import json
from typing import Any

from core.config import get_settings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from models.schemas import (
    CandidateEvaluation,
    CandidateEvaluationResult,
    Product,
    ProductEvidence,
)
from utils.json_utils import parse_json_object

from .base_agent import BaseAgent

SYSTEM_PROMPT = """你是电商候选商品评估Agent。请根据结构化购物需求和商品事实，逐个评估候选商品。

规则:
1. 硬约束不满足时 hard_constraints_passed 必须为 false；unmet_constraints 只记录未满足的硬约束。
2. fit_score 为0到100，表示商品与当前需求的匹配程度。
3. 只能使用输入中的商品字段作为证据；资料未提供的能力必须放入 unverified_requirements，不能推断为支持。
4. feedback 是上一轮校验问题；存在时必须针对问题修正评估。
5. 每个候选商品都要返回一条评估，不要添加输入中不存在的商品。
6. 只输出JSON对象，不要输出思维过程。

输出格式:
{
  "evaluations": [
    {
      "product_id": "商品ID",
      "fit_score": 85,
      "hard_constraints_passed": true,
      "matched_preferences": ["已满足的偏好"],
      "unmet_constraints": [],
      "unverified_requirements": ["资料未证实的要求"],
      "evidence": [{"field": "tags", "value": ["标签"]}],
      "reason": "简短、可核验的评估结论"
    }
  ]
}
"""

_EVIDENCE_FIELDS = {
    "product_id",
    "name",
    "category",
    "price",
    "brand",
    "stock",
    "tags",
    "description",
    "score",
}


class CandidateEvaluatorAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(
            name="candidate_evaluator",
            timeout=8.0,
            max_retries=1,
        )
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.0,
            max_tokens=1400,
        )

    async def _execute(self, **kwargs: Any) -> CandidateEvaluationResult:
        context: dict[str, Any] = kwargs.get("context", {})
        products: list[Product] = kwargs.get("products", [])
        feedback: list[dict[str, Any]] = kwargs.get("feedback", [])

        if not products:
            return CandidateEvaluationResult(
                success=True,
                evaluations=[],
                data={"evaluations": [], "feedback_applied": bool(feedback)},
                confidence=1.0,
            )

        payload = {
            "purchase_plan": self._purchase_plan(context),
            "products": [self._product_payload(product) for product in products],
            "feedback": feedback,
        }
        response = await self.llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        )
        data = parse_json_object(response.content)
        evaluations = self._clean_evaluations(data.get("evaluations"), products)
        coverage = len(evaluations) / len(products)

        return CandidateEvaluationResult(
            success=bool(evaluations),
            evaluations=evaluations,
            data={
                "evaluations": [item.model_dump() for item in evaluations],
                "feedback_applied": bool(feedback),
                "evaluated_count": len(evaluations),
                "candidate_count": len(products),
            },
            confidence=round(coverage, 2),
        )

    def _purchase_plan(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "query": str(context.get("query") or ""),
            "shopping_goals": context.get("shopping_goals") or [],
            "budget": context.get("budget"),
            "constraints": context.get("constraints") or [],
            "avoid_categories": context.get("avoid_categories") or [],
            "long_term_memory": context.get("long_term_memory") or [],
        }

    def _product_payload(self, product: Product) -> dict[str, Any]:
        return {
            "product_id": product.product_id,
            "name": product.name,
            "category": product.category,
            "price": product.price,
            "brand": product.brand,
            "stock": product.stock,
            "tags": product.tags,
            "description": product.description[:800],
            "score": product.score,
        }

    def _clean_evaluations(
        self,
        raw: Any,
        products: list[Product],
    ) -> list[CandidateEvaluation]:
        if not isinstance(raw, list):
            return []
        products_by_id = {product.product_id: product for product in products}
        evaluations: list[CandidateEvaluation] = []
        seen: set[str] = set()

        for item in raw:
            if not isinstance(item, dict):
                continue
            product_id = str(item.get("product_id") or "")
            product = products_by_id.get(product_id)
            if product is None or product_id in seen:
                continue
            seen.add(product_id)
            product_data = self._product_payload(product)
            evidence = self._clean_evidence(item.get("evidence"), product_data)
            evaluations.append(
                CandidateEvaluation(
                    product_id=product_id,
                    fit_score=self._fit_score(item.get("fit_score")),
                    hard_constraints_passed=(
                        item.get("hard_constraints_passed") is True
                    ),
                    matched_preferences=self._string_list(
                        item.get("matched_preferences")
                    ),
                    unmet_constraints=self._string_list(
                        item.get("unmet_constraints")
                    ),
                    unverified_requirements=self._string_list(
                        item.get("unverified_requirements")
                    ),
                    evidence=evidence,
                    reason=str(item.get("reason") or "").strip(),
                )
            )
        return evaluations

    def _clean_evidence(
        self,
        raw: Any,
        product_data: dict[str, Any],
    ) -> list[ProductEvidence]:
        if not isinstance(raw, list):
            return []
        evidence: list[ProductEvidence] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field") or "")
            if field not in _EVIDENCE_FIELDS or field in seen:
                continue
            seen.add(field)
            evidence.append(
                ProductEvidence(field=field, value=product_data.get(field))
            )
        return evidence

    def _fit_score(self, raw: Any) -> float:
        try:
            return min(100.0, max(0.0, float(raw)))
        except (TypeError, ValueError):
            return 0.0

    def _string_list(self, raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]
