"""Supervisor编排器 — 召回、库存过滤、导购表达聚合。"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import structlog

from agents import ProductRecAgent, ShoppingGuideAgent
from models.schemas import (
    Product,
    ProductRecResult,
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
        self.shopping_guide_agent = ShoppingGuideAgent()
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

        available_ids = await self._available_product_ids(raw_products)
        final_products = [p for p in raw_products if p.product_id in available_ids]
        final_products = final_products[:request.num_items]

        guide_query = str(
            request.context.get("query")
            or request.context.get("keyword")
            or request.scene
            or ""
        )
        if self.sales_rag_service is not None:
            augmented_context = await self.sales_rag_service.augment(
                user_message=guide_query,
                sales_context=request.context,
                products=[
                    product.model_dump()
                    for product in final_products
                ],
                memory=request.context.get("session_memory") or {},
            )
            request = request.model_copy(
                update={"context": augmented_context}
            )
        guide_result = await self.shopping_guide_agent.run(
            query=guide_query,
            context=request.context,
            products=final_products,
            requested_categories=rec_result.data.get("selected_categories", []),
            requested_keywords=rec_result.data.get("selected_keywords", []),
            image_summary=request.context.get("image_summary", ""),
        )
        copies = getattr(guide_result, "copies", [])

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
                "shopping_guide": guide_result,
            },
            rag_trace=request.context.get("rag_trace") or {},
            citations=request.context.get("citations") or [],
            total_latency_ms=total_latency,
        )

    async def _recommend_products(
        self,
        request: RecommendationRequest,
    ) -> ProductRecResult:
        goals = self._shopping_goals(request.context)
        if len(goals) <= 1:
            return await self.product_rec_agent.run(
                user_id=request.user_id,
                context=request.context,
                num_items=request.num_items * 2,
            )

        per_goal_items = max(1, (request.num_items * 2 + len(goals) - 1) // len(goals))
        goal_results = await asyncio.gather(
            *[
                self.product_rec_agent.run(
                    user_id=request.user_id,
                    context=self._context_for_goal(request.context, goal),
                    num_items=per_goal_items,
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
            recall_strategy="multi_goal",
            data={
                "candidate_count": len(merged_products),
                "reranked": len(merged_products),
                "recall_reason": "multi_goal_recommendation",
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

    async def _available_product_ids(self, products: list[Product]) -> set[str]:
        stock_map = await self.inventory_repository.get_stock_map(
            [product.product_id for product in products]
        )
        return {
            product.product_id
            for product in products
            if stock_map.get(product.product_id, 0) > 0
        }
