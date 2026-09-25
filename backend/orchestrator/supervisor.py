"""Supervisor编排器 — 召回、库存过滤、导购表达聚合。"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import structlog
from agents import ProductRecAgent
from agents.runtime import get_quality_reviewer, get_response_agent
from core.agent_config import get_agent_system_config
from services.candidate_evaluation import CandidateEvaluationService
from services.product_constraints import hard_constraint_failures
from models.schemas import (
    AgentResult,
    CandidateEvaluation,
    Product,
    ProductRecResult,
    QualityReviewResult,
    ShoppingGuideResult,
    RecommendationRequest,
    RecommendationResponse,
)
from repositories import InventoryRepository
from services.ab_test import ABTestEngine
from services.fusion.sales_rag_service import SalesRagService

logger = structlog.get_logger()


class SupervisorOrchestrator:
    """Coordinates recommendation, inventory, and guide generation."""

    def __init__(
        self,
        ab_engine: ABTestEngine | None = None,
        sales_rag_service: SalesRagService | None = None,
    ):
        self.product_rec_agent = ProductRecAgent()
        self.candidate_evaluation = CandidateEvaluationService()
        self.quality_reviewer = get_quality_reviewer()
        self.response_agent = get_response_agent()
        self.inventory_repository = InventoryRepository()
        self.ab_engine = ab_engine or ABTestEngine()
        self.sales_rag_service = sales_rag_service

    async def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        logger.info(
            "supervisor.start",
            request_id=request_id,
            user_id=request.user_id,
            scene=request.scene,
        )

        experiment = self.ab_engine.assign(request.user_id)
        copy_experiment = self.ab_engine.assign(request.user_id, "copy_style")
        request = request.model_copy(
            update={
                "context": {
                    **request.context,
                    "experiment_config": {
                        **(experiment.get("config") or {}),
                        **(copy_experiment.get("config") or {}),
                    },
                }
            }
        )

        rec_result = await self._recommend_products(request)
        raw_products: list[Product] = getattr(rec_result, "products", [])

        candidate_products = await self._filter_candidates(raw_products, request.context)

        guide_query = str(
            request.context.get("query")
            or request.context.get("keyword")
            or request.scene
            or ""
        )
        if self.sales_rag_service is not None and candidate_products:
            augmented_context = await self.sales_rag_service.augment(
                user_message=guide_query,
                sales_context=request.context,
                products=[
                    product.model_dump()
                    for product in candidate_products
                ],
                memory=request.context.get("session_memory") or {},
            )
            request = request.model_copy(
                update={"context": augmented_context}
            )

        evaluation_result, guide_result, final_products = await self._draft_response(
            request, candidate_products, rec_result,
        )
        audit_result = await self._review_response(
            request.context, final_products, guide_result,
        )
        initial_issues = self._audit_issues(audit_result)
        revision_count = 0
        if (
            get_agent_system_config().orchestration.max_replans > 0
            and audit_result.success
            and not self._audit_passed(audit_result)
            and audit_result.retry_recommended
        ):
            evaluation_result, guide_result, final_products = await self._draft_response(
                request, candidate_products, rec_result, feedback=initial_issues,
            )
            audit_result = await self._review_response(
                request.context, final_products, guide_result,
            )
            revision_count = 1
        audit_result.data.update({"revision_count": revision_count, "initial_issues": initial_issues})
        final_products, guide_result = self._reviewed_output(final_products, guide_result, audit_result)
        rec_result.data["candidate_evaluation"] = evaluation_result.data
        copies = guide_result.copies

        total_latency = (time.perf_counter() - start) * 1000

        logger.info(
            "supervisor.complete",
            request_id=request_id,
            total_latency_ms=round(total_latency, 1),
            product_count=len(final_products),
            copy_count=len(copies),
        )

        return RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            products=final_products,
            marketing_copies=copies,
            experiment_group=experiment.get("group", "control"),
            agent_results={
                "product_rec": rec_result,
                "quality_reviewer": audit_result,
                "response_generation": guide_result,
            },
            rag_trace=request.context.get("rag_trace") or {},
            citations=request.context.get("citations") or [],
            total_latency_ms=total_latency,
        )

    async def _filter_candidates(
        self, products: list[Product], context: dict[str, Any],
    ) -> list[Product]:
        stock_map = await self.inventory_repository.get_stock_map(
            [product.product_id for product in products]
        ) if products else {}
        current = [product.model_copy(update={"stock": stock_map.get(product.product_id, 0)}) for product in products]
        return [product for product in current if not hard_constraint_failures(product, context)]

    async def _draft_response(
        self, request: RecommendationRequest, candidates: list[Product],
        rec_result: ProductRecResult, feedback: list[dict[str, Any]] | None = None,
    ) -> tuple[AgentResult, ShoppingGuideResult, list[Product]]:
        evaluation = await self.candidate_evaluation.evaluate(
            context=request.context, products=candidates,
            strategy="semantic" if feedback else "auto", feedback=feedback or [],
        )
        if not evaluation.success:
            return evaluation, ShoppingGuideResult(
                success=False, error="候选评估未完成", confidence=0.0,
            ), []
        products = self._rank_evaluated_products(candidates, evaluation)[:request.num_items]
        products = [product for product in products if not hard_constraint_failures(product, request.context)]
        result = await self.response_agent.run(
            query=str(request.context.get("query") or request.context.get("keyword") or request.scene),
            context={**request.context, "candidate_evaluations": evaluation.data.get("evaluations", [])},
            products=products,
            requested_categories=rec_result.data.get("selected_categories", []),
            requested_keywords=rec_result.data.get("selected_keywords", []),
            image_summary=request.context.get("image_summary", ""),
            feedback=feedback or [],
        )
        if not isinstance(result, ShoppingGuideResult):
            result = ShoppingGuideResult(success=False, error=result.error, confidence=0.0)
        displayed = result.data.get("displayed_product_ids")
        if isinstance(displayed, list):
            products = [product for product in products if product.product_id in displayed]
        return evaluation, result, products

    async def _review_response(
        self, context: dict[str, Any], products: list[Product], draft: ShoppingGuideResult,
    ) -> QualityReviewResult:
        return await self.quality_reviewer.run(
            mode="recommendation",
            context={
                "query": context.get("query") or context.get("keyword") or "",
                "shopping_goals": context.get("shopping_goals") or [],
                "budget": context.get("budget"),
                "constraints": context.get("constraints") or [],
                "categories": context.get("categories") or [],
                "category": context.get("category"),
                "avoid_categories": context.get("avoid_categories") or [],
                "skip_category_filter": context.get("skip_category_filter", False),
                "image_summary": context.get("image_summary", ""),
                "available_product_count": len(products),
            },
            products=products,
            evidence=context.get("rag_context") or [],
            draft={
                "answer": draft.answer if draft.success else "",
                "product_copies": draft.copies,
                "decision_summary": draft.data.get("decision_summary") or {},
                "displayed_products": [product.model_dump() for product in products],
            },
        )

    def _reviewed_output(
        self, products: list[Product], draft: ShoppingGuideResult, audit: AgentResult,
    ) -> tuple[list[Product], ShoppingGuideResult]:
        if self._audit_passed(audit) and draft.success:
            return products, draft
        return [], ShoppingGuideResult(
            success=True,
            answer="当前资料还不足以给出可靠建议，请补充商品或需求信息后再试。",
            copies=[],
            data={"displayed_product_ids": [], "review_blocked": True},
            confidence=0.0,
        )

    def _evaluations(
        self,
        result: AgentResult,
    ) -> list[CandidateEvaluation]:
        evaluations = getattr(result, "evaluations", [])
        return [
            item for item in evaluations if isinstance(item, CandidateEvaluation)
        ]

    def _audit_issues(self, result: AgentResult) -> list[dict[str, Any]]:
        issues = getattr(result, "issues", [])
        return [
            item.model_dump()
            for item in issues
            if hasattr(item, "model_dump")
        ]

    def _audit_passed(self, result: AgentResult) -> bool:
        return bool(result.success and getattr(result, "passed", False))

    def _rank_evaluated_products(
        self,
        products: list[Product],
        result: AgentResult,
    ) -> list[Product]:
        evaluations = {
            item.product_id: item
            for item in self._evaluations(result)
        }
        indexed = list(enumerate(products))
        eligible = [
            (index, product, evaluations[product.product_id])
            for index, product in indexed
            if product.product_id in evaluations
            and evaluations[product.product_id].hard_constraints_passed
        ]
        eligible.sort(key=lambda item: (-item[2].fit_score, item[0]))
        return [product for _, product, _ in eligible]

    async def _recommend_products(
        self,
        request: RecommendationRequest,
        *,
        recall_mode: str = "hybrid",
    ) -> ProductRecResult:
        goals = self._shopping_goals(request.context)
        if len(goals) <= 1:
            return await self.product_rec_agent.run(
                user_id=request.user_id,
                context=request.context,
                num_items=request.num_items * 2,
                recall_mode=recall_mode,
            )

        per_goal_items = max(1, (request.num_items * 2 + len(goals) - 1) // len(goals))
        goal_results = await asyncio.gather(
            *[
                self.product_rec_agent.run(
                    user_id=request.user_id,
                    context=self._context_for_goal(request.context, goal),
                    num_items=per_goal_items,
                    recall_mode=recall_mode,
                )
                for goal in goals
            ]
        )
        selected_categories: list[str] = []
        for result in goal_results:
            for category in result.data.get("selected_categories", []):
                if category and category not in selected_categories:
                    selected_categories.append(category)

        merged_products = self._merge_goal_products(
            goal_results,
            limit=request.num_items * 2,
        )

        return ProductRecResult(
            success=all(result.success for result in goal_results),
            products=merged_products[: request.num_items * 2],
            recall_strategy=f"multi_goal_{recall_mode}",
            data={
                "candidate_count": len(merged_products),
                "reranked": len(merged_products),
                "recall_reason": f"multi_goal_{recall_mode}_recommendation",
                "goal_count": len(goals),
                "goal_results": [
                    {
                        "goal": goals[index],
                        "product_count": len(getattr(result, "products", [])),
                        "selected_categories": result.data.get("selected_categories", []),
                        "query_recall_count": result.data.get("query_recall_count", 0),
                        "profile_recall_count": result.data.get("profile_recall_count", 0),
                    }
                    for index, result in enumerate(goal_results)
                ],
                "selected_categories": selected_categories,
                "selected_keywords": [],
            },
            confidence=min((result.confidence for result in goal_results), default=0.8),
        )

    def _merge_goal_products(
        self,
        goal_results: list[ProductRecResult],
        limit: int,
    ) -> list[Product]:
        merged_products: list[Product] = []
        seen: set[str] = set()
        product_groups = [
            list(getattr(result, "products", []))
            for result in goal_results
        ]
        max_group_size = max((len(group) for group in product_groups), default=0)

        for index in range(max_group_size):
            for group in product_groups:
                if index >= len(group):
                    continue
                product = group[index]
                if product.product_id in seen:
                    continue
                merged_products.append(product)
                seen.add(product.product_id)
                if len(merged_products) >= limit:
                    return merged_products

        return merged_products

    def _shopping_goals(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        raw_goals = context.get("shopping_goals")
        if isinstance(raw_goals, list):
            goals = [goal for goal in raw_goals if isinstance(goal, dict)]
            if goals:
                return goals
        categories = context.get("categories")
        if isinstance(categories, list) and len(categories) > 1:
            return [
                {
                    "goal": str(context.get("query") or category),
                    "category": str(category),
                    "keywords": [],
                    "constraints": context.get("constraints") or [],
                }
                for category in categories
                if category
            ]
        return []

    def _context_for_goal(
        self,
        context: dict[str, Any],
        goal: dict[str, Any],
    ) -> dict[str, Any]:
        category = str(goal.get("category") or "")
        goal_query = " ".join(
            str(part)
            for part in [
                goal.get("goal"),
                category,
                " ".join(str(item) for item in goal.get("keywords", []) if item)
                if isinstance(goal.get("keywords"), list)
                else "",
                " ".join(str(item) for item in goal.get("constraints", []) if item)
                if isinstance(goal.get("constraints"), list)
                else "",
            ]
            if part
        ).strip()
        goal_context = {
            **context,
            "query": goal_query or str(context.get("query") or ""),
            "shopping_goals": [goal],
        }
        if category:
            goal_context["categories"] = [category]
        return goal_context
