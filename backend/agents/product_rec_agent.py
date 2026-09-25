"""
商品推荐Agent
- 召回层：协同过滤 + 向量检索(Milvus) + 热度/新品策略
- 排序层：召回分数 + 上下文过滤
- 多样性控制：类目打散、卖家去重、新品加权
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal
from services.product_constraints import hard_constraint_failures

from core.agent_config import get_agent_system_config
from models.schemas import Product, ProductRecResult
from services.recall import VectorRecallService

from .base_agent import BaseAgent


@dataclass
class _RecallOutcome:
    products: list[Product]
    strategy: str
    reason: str
    query: str
    categories: list[str]
    keywords: list[str]
    query_count: int
    profile_count: int


@dataclass(frozen=True)
class RecallScenarioBid:
    can_handle: bool
    scenario_score: float
    reason: str


RecallMode = Literal["hybrid", "semantic", "profile", "popularity"]


class ProductRecAgent(BaseAgent):
    def __init__(self):
        definition = get_agent_system_config().agent("product-recommendation")
        super().__init__(
            name="product_rec",
            timeout=definition.runtime.timeout_seconds,
            max_retries=definition.runtime.max_attempts,
        )
        self.vector_recall_service = VectorRecallService()

    @staticmethod
    def bid_recall_mode(
        mode: RecallMode,
        context: dict[str, Any],
    ) -> RecallScenarioBid:
        """Let each recall strategy assess its own fit before execution."""
        query = str(
            context.get("query")
            or context.get("keyword")
            or context.get("image_query")
            or ""
        ).strip()
        has_profile = bool(
            context.get("long_term_memory")
            or context.get("user_profile")
            or context.get("behavior_profile")
        )
        goals = context.get("shopping_goals") or []

        if mode == "semantic":
            if len(goals) > 1:
                return RecallScenarioBid(
                    can_handle=True,
                    scenario_score=0.9,
                    reason="存在多个购物目标，语义召回适合综合理解需求",
                )
            if len(query) > 6:
                return RecallScenarioBid(
                    can_handle=True,
                    scenario_score=1.0,
                    reason="查询描述具体，语义召回适合处理明确约束",
                )
            if query:
                return RecallScenarioBid(
                    can_handle=True,
                    scenario_score=0.65,
                    reason="存在简短查询，语义召回可以执行但匹配优势有限",
                )
            return RecallScenarioBid(
                can_handle=True,
                scenario_score=0.3,
                reason="缺少明确查询，语义召回的场景匹配度较低",
            )

        if mode == "profile":
            if has_profile:
                return RecallScenarioBid(
                    can_handle=True,
                    scenario_score=1.0,
                    reason="存在用户画像或长期记忆，适合画像召回",
                )
            return RecallScenarioBid(
                can_handle=True,
                scenario_score=0.25,
                reason="当前请求没有可用画像信号，画像召回匹配度较低",
            )

        if mode == "popularity":
            if not has_profile and len(query) <= 6:
                return RecallScenarioBid(
                    can_handle=True,
                    scenario_score=1.0,
                    reason="冷启动且需求宽泛，热门召回适合作为首选",
                )
            if not has_profile:
                return RecallScenarioBid(
                    can_handle=True,
                    scenario_score=0.45,
                    reason="缺少用户画像，但具体查询更适合语义召回",
                )
            return RecallScenarioBid(
                can_handle=True,
                scenario_score=0.3,
                reason="已有画像信号，热门召回仅作为通用候选来源",
            )

        return RecallScenarioBid(
            can_handle=True,
            scenario_score=0.8,
            reason="混合召回能够覆盖当前请求",
        )

    @classmethod
    def select_recall_mode(cls, context: dict[str, Any]) -> tuple[RecallMode, str]:
        """Choose a strategy inside one recall agent using the existing scenario bids."""
        modes: tuple[RecallMode, ...] = ("semantic", "profile", "popularity")
        bids = [(mode, cls.bid_recall_mode(mode, context)) for mode in modes]
        eligible = [(mode, bid) for mode, bid in bids if bid.can_handle]
        if not eligible:
            return "hybrid", "没有适用的单路策略，使用混合召回"
        mode, bid = max(eligible, key=lambda item: item[1].scenario_score)
        return mode, bid.reason

    async def _execute(self, **kwargs: Any) -> ProductRecResult:
        user_id: str = kwargs.get("user_id", "")
        num_items: int = kwargs.get("num_items", 10)
        context: dict[str, Any] = kwargs.get("context", {})
        recall_mode = self._recall_mode(kwargs.get("recall_mode"))

        recall = await self._recall(
            user_id,
            context,
            num_items * 3,
            mode=recall_mode,
        )
        candidates = recall.products
        candidates = self._apply_context_filters(candidates, context)
        if not candidates:
            return ProductRecResult(
                success=True,
                products=[],
                recall_strategy=recall.strategy,
                data={
                    "candidate_count": 0,
                    "reranked": 0,
                    "recall_reason": recall.reason,
                    "recall_query": recall.query,
                    "recall_categories": recall.categories,
                    "query_recall_count": recall.query_count,
                    "profile_recall_count": recall.profile_count,
                    "selected_categories": recall.categories,
                    "selected_keywords": recall.keywords,
                },
                confidence=1.0,
            )

        experiment_config = context.get("experiment_config") or {}
        rerank_strategy = (
            experiment_config.get("rerank", "diversified")
            if isinstance(experiment_config, dict)
            else "diversified"
        )
        ranked_ids = self._rank(candidates, num_items, rerank_strategy)

        id_to_product = {p.product_id: p for p in candidates}
        final_products = []
        for pid in ranked_ids:
            if pid in id_to_product:
                final_products.append(id_to_product[pid])
        if len(final_products) < num_items:
            for p in candidates:
                if p.product_id not in ranked_ids:
                    final_products.append(p)
                    if len(final_products) >= num_items:
                        break

        return ProductRecResult(
            success=True,
            products=final_products[:num_items],
            recall_strategy=recall.strategy,
            data={
                "candidate_count": len(candidates),
                "reranked": len(ranked_ids),
                "recall_reason": recall.reason,
                "recall_query": recall.query,
                "recall_categories": recall.categories,
                "query_recall_count": recall.query_count,
                "profile_recall_count": recall.profile_count,
                "selected_categories": recall.categories,
                "selected_keywords": recall.keywords,
                "rerank_strategy": rerank_strategy,
            },
            confidence=0.8,
        )

    async def _recall(
        self,
        user_id: str,
        context: dict[str, Any],
        limit: int,
        *,
        mode: RecallMode = "hybrid",
    ) -> _RecallOutcome:
        """Execute one broker-selectable recall strategy with safe fallbacks."""
        query = self._build_query(context)
        categories = await self._resolve_categories(
            query=query,
            categories=self._context_categories(context),
            context=context,
        )
        keywords = self._resolve_keywords(query)

        if mode == "semantic":
            query_products = await self.vector_recall_service.recall_by_query(
                query=query,
                profile=None,
                categories=categories,
                keywords=keywords,
                limit=limit,
            )
            products = await self._complete_with_popularity(
                query_products,
                categories=categories,
                limit=limit,
            )
            return _RecallOutcome(
                products=products,
                strategy="semantic_embedding",
                reason="selected_semantic_recall",
                query=query,
                categories=categories,
                keywords=keywords,
                query_count=len(query_products),
                profile_count=0,
            )

        if mode == "profile":
            profile_products = await self.vector_recall_service.recall_by_profile(
                user_id=user_id,
                profile=None,
                limit=limit,
            )
            if categories:
                profile_products = self._keep_requested_categories(
                    profile_products,
                    categories,
                )
            products = await self._complete_with_popularity(
                profile_products,
                categories=categories,
                limit=limit,
            )
            return _RecallOutcome(
                products=products,
                strategy="profile_swing",
                reason="selected_profile_recall",
                query=query,
                categories=categories,
                keywords=keywords,
                query_count=0,
                profile_count=len(profile_products),
            )

        if mode == "popularity":
            products = (
                await self.vector_recall_service.product_repository.recall_candidates(
                    profile=None,
                    categories=categories or None,
                    limit=limit,
                )
            )
            return _RecallOutcome(
                products=products,
                strategy="popularity",
                reason="selected_popularity_recall",
                query=query,
                categories=categories,
                keywords=keywords,
                query_count=0,
                profile_count=0,
            )

        query_products, profile_products = await asyncio.gather(
            self.vector_recall_service.recall_by_query(
                query=query,
                profile=None,
                categories=categories,
                keywords=keywords,
                limit=limit,
            ),
            self.vector_recall_service.recall_by_profile(
                user_id=user_id,
                profile=None,
                limit=limit,
            ),
        )
        merged = self.vector_recall_service.merge_ranked_products(
            query_products=query_products,
            user_products=profile_products,
            limit=limit,
        )
        if categories:
            merged = self._keep_requested_categories(merged, categories)
            if len(merged) < limit:
                fallback_products = await self.vector_recall_service.product_repository.recall_candidates(
                    profile=None,
                    categories=categories,
                    limit=limit,
                )
                merged = self._append_unique(merged, fallback_products, limit)
        elif not merged:
            merged = (
                await self.vector_recall_service.product_repository.recall_candidates(
                    profile=None,
                    categories=None,
                    limit=limit,
                )
            )
        return _RecallOutcome(
            products=merged,
            strategy="swing_embedding",
            reason="fixed_query_embedding_and_swing_recall",
            query=query,
            categories=categories,
            keywords=keywords,
            query_count=len(query_products),
            profile_count=len(profile_products),
        )

    def _recall_mode(self, value: Any) -> RecallMode:
        if value in {"hybrid", "semantic", "profile", "popularity"}:
            return value
        return "hybrid"

    async def _complete_with_popularity(
        self,
        products: list[Product],
        *,
        categories: list[str],
        limit: int,
    ) -> list[Product]:
        if len(products) >= limit:
            return products[:limit]
        fallback_products = (
            await self.vector_recall_service.product_repository.recall_candidates(
                profile=None,
                categories=categories or None,
                limit=limit,
            )
        )
        return self._append_unique(products, fallback_products, limit)

    def _build_query(self, context: dict[str, Any]) -> str:
        parts = [
            context.get("query"),
            context.get("keyword"),
            context.get("image_query"),
            context.get("image_summary"),
            self._join_text(context.get("image_attributes")),
            self._shopping_goals_text(context.get("shopping_goals")),
            self._join_text(context.get("categories")),
            context.get("category"),
            context.get("image_category"),
            self._long_term_memory_text(context.get("long_term_memory")),
        ]
        return " ".join(str(part).strip() for part in parts if part).strip()

    def _context_categories(self, context: dict[str, Any]) -> list[str]:
        if context.get("skip_category_filter"):
            return []
        goal_categories = self._shopping_goal_categories(context.get("shopping_goals"))
        raw = (
            context.get("categories")
            or context.get("category")
            or context.get("image_category")
            or []
        )
        categories = []
        if isinstance(raw, str):
            categories = [raw] if raw else []
        elif isinstance(raw, list):
            categories = [str(item) for item in raw if item]
        return self._merge_values(categories, goal_categories)

    def _shopping_goal_categories(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        categories = []
        for item in value:
            if isinstance(item, dict) and item.get("category"):
                categories.append(str(item["category"]))
        return categories

    def _merge_values(self, *value_groups: list[str]) -> list[str]:
        merged = []
        for values in value_groups:
            for value in values:
                if value and value not in merged:
                    merged.append(value)
        return merged

    def _shopping_goals_text(self, value: Any) -> str:
        if not isinstance(value, list):
            return ""
        parts = []
        for item in value:
            if not isinstance(item, dict):
                continue
            goal = str(item.get("goal") or "")
            category = str(item.get("category") or "")
            keywords = self._join_text(item.get("keywords"))
            constraints = self._join_text(item.get("constraints"))
            text = " ".join(part for part in [goal, category, keywords, constraints] if part)
            if text:
                parts.append(text)
        return " ".join(parts)

    def _join_text(self, value: Any) -> str:
        if isinstance(value, list):
            return " ".join(str(item) for item in value if item)
        return str(value or "").strip()

    def _long_term_memory_text(self, value: Any) -> str:
        if not isinstance(value, list):
            return ""
        parts = []
        for item in value:
            if not isinstance(item, dict):
                continue
            memory_type = str(item.get("type") or "")
            memory_value = str(item.get("value") or "")
            if memory_value:
                parts.append(f"{memory_type} {memory_value}".strip())
        return " ".join(parts)

    async def _resolve_categories(
        self,
        query: str,
        categories: list[str],
        context: dict[str, Any],
    ) -> list[str]:
        available = await self.vector_recall_service.product_repository.list_active_categories()
        available_set = set(available)
        resolved = [category for category in categories if category in available_set]

        for category in available:
            if category and category in query and category not in resolved:
                resolved.append(category)

        raw_category = context.get("category")
        if isinstance(raw_category, str) and raw_category in available_set:
            if raw_category not in resolved:
                resolved.append(raw_category)

        raw_categories = context.get("categories")
        if isinstance(raw_categories, list):
            for category in raw_categories:
                category = str(category)
                if category in available_set and category not in resolved:
                    resolved.append(category)

        return resolved

    def _resolve_keywords(self, query: str) -> list[str]:
        feature_terms = (
            "降噪",
            "通勤",
            "无线",
            "蓝牙",
            "续航",
            "快充",
            "轻便",
            "便携",
            "防水",
            "运动",
            "游戏",
            "办公",
            "学习",
            "护眼",
            "静音",
            "大容量",
            "低糖",
        )
        normalized = query.lower()
        keywords = [term for term in feature_terms if term in normalized]
        for token in normalized.replace("/", " ").replace("-", " ").split():
            cleaned = token.strip("，。！？,.!?()（）")
            # 中文查询通常不含空格，整句会被切成单一 token；原句不是关键词，
            # 直接跳过，避免下游用整句子串匹配把召回结果全部过滤掉。
            if (
                len(cleaned) >= 2
                and cleaned != normalized
                and cleaned not in keywords
            ):
                keywords.append(cleaned)
        return keywords[:8]

    def _keep_requested_categories(
        self,
        products: list[Product],
        categories: list[str],
    ) -> list[Product]:
        selected = set(categories)
        return [product for product in products if product.category in selected]

    def _append_unique(
        self,
        products: list[Product],
        extra_products: list[Product],
        limit: int,
    ) -> list[Product]:
        seen = {product.product_id for product in products}
        merged = list(products)
        for product in extra_products:
            if product.product_id in seen:
                continue
            merged.append(product)
            seen.add(product.product_id)
            if len(merged) >= limit:
                break
        return merged

    def _apply_context_filters(
        self,
        products: list[Product],
        context: dict[str, Any],
    ) -> list[Product]:
        return [product for product in products if not hard_constraint_failures(product, context)]

    def _rank(
        self,
        candidates: list[Product],
        num_items: int,
        strategy: str = "diversified",
    ) -> list[str]:
        indexed = list(enumerate(candidates))
        indexed.sort(
            key=lambda item: (
                -float(item[1].score or 0.0),
                item[0],
            )
        )
        remaining = [product for _, product in indexed]
        if strategy == "rule_based":
            return [product.product_id for product in remaining[:num_items]]

        ranked: list[Product] = []
        used_sellers: set[str] = set()
        last_category = ""

        while remaining and len(ranked) < num_items:
            selected_index = 0
            for index, product in enumerate(remaining):
                seller_is_new = (
                    not product.seller_id
                    or product.seller_id not in used_sellers
                )
                category_is_new = product.category != last_category
                if seller_is_new and category_is_new:
                    selected_index = index
                    break
            product = remaining.pop(selected_index)
            ranked.append(product)
            if product.seller_id:
                used_sellers.add(product.seller_id)
            last_category = product.category

        return [product.product_id for product in ranked]
