from __future__ import annotations

import json
from typing import Any

from core.config import get_settings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from models.schemas import (
    CandidateEvaluation,
    Product,
    RecommendationAuditResult,
    RecommendationIssue,
)
from utils.json_utils import parse_json_object

from .base_agent import BaseAgent

SYSTEM_PROMPT = """你是电商推荐结果校验Agent。请检查候选评估是否遵守用户需求和商品事实。

重点检查:
1. 是否把不满足预算、类目、库存或其他硬约束的商品标记为通过。
2. 推荐理由和匹配偏好是否能由商品事实支持。
3. 是否把未知参数误写成商品能力。
4. fit_score 和推荐顺序是否与需求匹配程度明显矛盾。

只有发现会影响推荐正确性的具体问题时才建议重评估。不要要求披露思维过程。
只输出JSON对象:
{
  "passed": true,
  "retry_recommended": false,
  "issues": [
    {
      "product_id": "商品ID或空字符串",
      "issue_type": "hard_constraint|unsupported_claim|ranking_conflict|missing_evaluation",
      "detail": "可执行的修正意见"
    }
  ]
}
"""


class RecommendationCriticAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(
            name="recommendation_critic",
            timeout=6.0,
            max_retries=1,
        )
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.0,
            max_tokens=700,
        )

    async def _execute(self, **kwargs: Any) -> RecommendationAuditResult:
        context: dict[str, Any] = kwargs.get("context", {})
        products: list[Product] = kwargs.get("products", [])
        evaluations: list[CandidateEvaluation] = kwargs.get("evaluations", [])
        deterministic_issues = self._deterministic_issues(
            context=context,
            products=products,
            evaluations=evaluations,
        )

        payload = {
            "purchase_plan": {
                "query": str(context.get("query") or ""),
                "shopping_goals": context.get("shopping_goals") or [],
                "budget": context.get("budget"),
                "constraints": context.get("constraints") or [],
                "avoid_categories": context.get("avoid_categories") or [],
            },
            "products": [self._product_payload(product) for product in products],
            "evaluations": [item.model_dump() for item in evaluations],
            "deterministic_issues": [
                item.model_dump() for item in deterministic_issues
            ],
        }
        response = await self.llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        )
        data = parse_json_object(response.content)
        if not isinstance(data.get("passed"), bool) or not isinstance(
            data.get("issues", []), list
        ):
            raise ValueError("推荐校验Agent未返回有效的结构化结果")
        model_issues = self._clean_issues(data.get("issues"), products)
        issues = self._merge_issues(deterministic_issues, model_issues)
        passed = not issues and data["passed"] is True
        retry_recommended = bool(issues) and (
            data.get("retry_recommended", True) is True
        )

        return RecommendationAuditResult(
            success=True,
            passed=passed,
            retry_recommended=retry_recommended,
            issues=issues,
            data={
                "passed": passed,
                "retry_recommended": retry_recommended,
                "issues": [item.model_dump() for item in issues],
            },
            confidence=0.9 if passed else 0.75,
        )

    def _deterministic_issues(
        self,
        context: dict[str, Any],
        products: list[Product],
        evaluations: list[CandidateEvaluation],
    ) -> list[RecommendationIssue]:
        if products and not evaluations:
            return [
                RecommendationIssue(
                    issue_type="missing_evaluation",
                    detail="候选商品缺少结构化评估，无法验证推荐依据",
                )
            ]

        products_by_id = {product.product_id: product for product in products}
        evaluated_ids = {item.product_id for item in evaluations}
        issues: list[RecommendationIssue] = []
        budget = self._budget(context.get("budget"))
        avoided = {
            str(item)
            for item in (context.get("avoid_categories") or [])
            if item
        }

        for product in products:
            if product.product_id not in evaluated_ids:
                issues.append(
                    RecommendationIssue(
                        product_id=product.product_id,
                        issue_type="missing_evaluation",
                        detail="该候选商品未被评估",
                    )
                )

        for evaluation in evaluations:
            product = products_by_id.get(evaluation.product_id)
            if product is None:
                continue
            hard_failures = []
            if budget is not None and product.price > budget:
                hard_failures.append(f"价格{product.price:g}超过预算{budget:g}")
            if product.stock <= 0:
                hard_failures.append("库存不足")
            if product.category in avoided:
                hard_failures.append(f"类目{product.category}已被排除")
            if evaluation.unmet_constraints:
                hard_failures.extend(evaluation.unmet_constraints)
            if hard_failures and evaluation.hard_constraints_passed:
                issues.append(
                    RecommendationIssue(
                        product_id=product.product_id,
                        issue_type="hard_constraint",
                        detail="；".join(dict.fromkeys(hard_failures)),
                    )
                )
        return issues

    def _clean_issues(
        self,
        raw: Any,
        products: list[Product],
    ) -> list[RecommendationIssue]:
        if not isinstance(raw, list):
            return []
        product_ids = {product.product_id for product in products}
        issues: list[RecommendationIssue] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            product_id = str(item.get("product_id") or "")
            if product_id and product_id not in product_ids:
                continue
            issue_type = str(item.get("issue_type") or "").strip()
            detail = str(item.get("detail") or "").strip()
            if not issue_type or not detail:
                continue
            issues.append(
                RecommendationIssue(
                    product_id=product_id,
                    issue_type=issue_type,
                    detail=detail,
                )
            )
        return issues

    def _merge_issues(
        self,
        *issue_groups: list[RecommendationIssue],
    ) -> list[RecommendationIssue]:
        merged: list[RecommendationIssue] = []
        seen: set[tuple[str, str, str]] = set()
        for issues in issue_groups:
            for issue in issues:
                key = (issue.product_id, issue.issue_type, issue.detail)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(issue)
        return merged

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
        }

    def _budget(self, raw: Any) -> float | None:
        try:
            budget = float(raw)
        except (TypeError, ValueError):
            return None
        return budget if budget > 0 else None
