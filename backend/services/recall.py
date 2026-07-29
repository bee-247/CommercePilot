from __future__ import annotations

import math

from models.schemas import Product, UserProfile
from repositories import ProductRepository, UserRepository
from services.embedding import EmbeddingService
from services.vector_store import MilvusVectorStore


class VectorRecallService:
    """Runs query embedding and profile recall, then hydrates product details."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = MilvusVectorStore()
        self.product_repository = ProductRepository()
        self.user_repository = UserRepository()

    async def recall_by_query(
        self,
        query: str,
        profile: UserProfile | None,
        categories: list[str],
        limit: int,
        keywords: list[str] | None = None,
    ) -> list[Product]:
        query = query.strip()
        keywords = keywords or []
        if not query:
            return []

        query_vector = await self.embedding_service.embed_text(query)
        if not query_vector:
            return []

        search_limit = limit * 5 if categories else limit
        product_ids = await self.vector_store.search_products(query_vector, search_limit)
        products = await self.product_repository.get_products_by_ids(
            product_ids=product_ids,
            profile=profile,
            categories=categories or None,
        )
        if keywords:
            keyword_products = self._filter_by_keywords(products, keywords)
            if keyword_products:
                return keyword_products[:limit]
            keyword_products = await self.product_repository.search_products_by_keywords(
                keywords=keywords,
                profile=profile,
                categories=categories or None,
                limit=limit,
            )
            if keyword_products:
                return keyword_products
        if products:
            return products[:limit]
        if categories:
            return await self.product_repository.recall_candidates(
                profile=profile,
                categories=categories,
                limit=limit,
            )
        return []

    async def recall_by_profile(
        self,
        user_id: str,
        profile: UserProfile | None,
        limit: int,
    ) -> list[Product]:
        products = await self._recall_by_swing(
            user_id=user_id,
            profile=profile,
            limit=limit,
        )
        if products:
            return products
        return await self._recall_by_user_embedding(
            user_id=user_id,
            profile=profile,
            limit=limit,
        )

    def merge_ranked_products(
        self,
        query_products: list[Product],
        user_products: list[Product],
        limit: int,
        query_weight: float = 0.6,
        user_weight: float = 0.4,
    ) -> list[Product]:
        products: dict[str, Product] = {}
        scores: dict[str, float] = {}

        for rank, product in enumerate(query_products):
            products[product.product_id] = product
            scores[product.product_id] = (
                scores.get(product.product_id, 0.0)
                + query_weight / (rank + 1)
            )

        for rank, product in enumerate(user_products):
            products[product.product_id] = product
            scores[product.product_id] = (
                scores.get(product.product_id, 0.0)
                + user_weight / (rank + 1)
            )

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        merged = []
        for product_id in ranked_ids[:limit]:
            product = products[product_id]
            product.score = scores[product_id]
            merged.append(product)
        return merged

    async def _recall_by_user_embedding(
        self,
        user_id: str,
        profile: UserProfile | None,
        limit: int,
    ) -> list[Product]:
        user_embedding = await self.vector_store.get_user_embedding(user_id)
        if not user_embedding:
            return []
        product_ids = await self.vector_store.search_products(user_embedding, limit)
        return await self.product_repository.get_products_by_ids(
            product_ids=product_ids,
            profile=profile,
        )

    async def _recall_by_swing(
        self,
        user_id: str,
        profile: UserProfile | None,
        limit: int,
    ) -> list[Product]:
        recent = await self.user_repository.get_recent_behaviors(user_id, limit=20)
        seed_scores = {
            str(row["product_id"]): float(row.get("weight") or 1.0)
            for row in recent
            if row.get("product_id")
        }
        if not seed_scores:
            return []

        sequences = await self.user_repository.list_behavior_sequences(limit=5000)
        item_users: dict[str, dict[str, float]] = {}
        for history_user_id, items in sequences.items():
            seen_for_user: dict[str, float] = {}
            for product_id, weight in items:
                seen_for_user[product_id] = max(
                    seen_for_user.get(product_id, 0.0),
                    weight,
                )
            for product_id, weight in seen_for_user.items():
                item_users.setdefault(product_id, {})[history_user_id] = weight

        scores: dict[str, float] = {}
        for seed_id, seed_weight in seed_scores.items():
            seed_users = item_users.get(seed_id, {})
            if not seed_users:
                continue
            for candidate_id, candidate_users in item_users.items():
                if candidate_id == seed_id or candidate_id in seed_scores:
                    continue
                overlap = set(seed_users).intersection(candidate_users)
                if not overlap:
                    continue
                swing_score = 0.0
                overlap_users = list(overlap)
                for i, user_a in enumerate(overlap_users):
                    for user_b in overlap_users[i + 1:]:
                        common_count = self._common_item_count(
                            sequences.get(user_a, []),
                            sequences.get(user_b, []),
                        )
                        swing_score += 1.0 / (1.0 + math.sqrt(common_count))
                if swing_score:
                    scores[candidate_id] = scores.get(candidate_id, 0.0) + (
                        seed_weight * swing_score
                    )

        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
        products = await self.product_repository.get_products_by_ids(
            product_ids=ranked_ids,
            profile=profile,
        )
        for product in products:
            product.score = scores.get(product.product_id, product.score)
        return products

    def _common_item_count(
        self,
        items_a: list[tuple[str, float]],
        items_b: list[tuple[str, float]],
    ) -> int:
        item_ids_a = {product_id for product_id, _ in items_a}
        item_ids_b = {product_id for product_id, _ in items_b}
        return len(item_ids_a.intersection(item_ids_b))

    def _filter_by_keywords(
        self,
        products: list[Product],
        keywords: list[str],
    ) -> list[Product]:
        filtered = []
        for product in products:
            text = " ".join(
                [
                    product.name,
                    product.category,
                    " ".join(product.tags),
                ]
            )
            if any(keyword and keyword in text for keyword in keywords):
                filtered.append(product)
        return filtered
