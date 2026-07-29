from __future__ import annotations

import json
from typing import Iterable

from sqlalchemy import desc, distinct, or_, select

from database.models import InventoryRecord, ProductRecord
from database.session import ProductSessionLocal
from models.schemas import Product, UserProfile


class ProductRepository:
    """Database-backed product recall repository."""

    async def recall_candidates(
        self,
        profile: UserProfile | None = None,
        categories: list[str] | None = None,
        limit: int = 30,
    ) -> list[Product]:
        with ProductSessionLocal() as session:
            stmt = self._base_product_stmt()
            stmt = self._apply_hard_filters(stmt, profile, categories)
            stmt = stmt.order_by(desc(ProductRecord.hot_score), desc(ProductRecord.created_at))
            rows = session.execute(stmt.limit(limit)).all()

        return [self._to_product(record, stock) for record, stock in rows]

    async def get_products_by_ids(
        self,
        product_ids: list[str],
        profile: UserProfile | None = None,
        categories: list[str] | None = None,
    ) -> list[Product]:
        if not product_ids:
            return []

        with ProductSessionLocal() as session:
            stmt = self._base_product_stmt().where(ProductRecord.product_id.in_(product_ids))
            stmt = self._apply_hard_filters(stmt, profile, categories)
            rows = session.execute(stmt).all()

        by_id = {record.product_id: self._to_product(record, stock) for record, stock in rows}
        return [by_id[product_id] for product_id in product_ids if product_id in by_id]

    async def list_products_for_index(self, limit: int = 1000) -> list[Product]:
        with ProductSessionLocal() as session:
            rows = session.execute(
                self._base_product_stmt()
                .order_by(desc(ProductRecord.updated_at))
                .limit(limit)
            ).all()
        return [self._to_product(record, stock) for record, stock in rows]

    async def search_products_by_keywords(
        self,
        keywords: list[str],
        profile: UserProfile | None = None,
        categories: list[str] | None = None,
        limit: int = 30,
    ) -> list[Product]:
        keywords = self._clean_values(keywords)
        if not keywords:
            return []

        with ProductSessionLocal() as session:
            stmt = self._base_product_stmt()
            stmt = self._apply_hard_filters(stmt, profile, categories)
            conditions = []
            for keyword in keywords:
                pattern = f"%{keyword}%"
                conditions.extend(
                    [
                        ProductRecord.name.like(pattern),
                        ProductRecord.category.like(pattern),
                        ProductRecord.tags_json.like(pattern),
                    ]
                )
            stmt = stmt.where(or_(*conditions))
            rows = session.execute(
                stmt.order_by(desc(ProductRecord.hot_score), desc(ProductRecord.created_at))
                .limit(limit)
            ).all()
        return [self._to_product(record, stock) for record, stock in rows]

    async def list_active_categories(self) -> list[str]:
        with ProductSessionLocal() as session:
            rows = session.execute(
                select(distinct(ProductRecord.category))
                .where(ProductRecord.status == "active")
                .order_by(ProductRecord.category)
            ).all()
        return [category for (category,) in rows if category]

    def _base_product_stmt(self):
        return (
            select(ProductRecord, InventoryRecord.stock)
            .outerjoin(
                InventoryRecord,
                InventoryRecord.product_id == ProductRecord.product_id,
            )
            .where(ProductRecord.status == "active")
        )

    def _apply_hard_filters(
        self,
        stmt,
        profile: UserProfile | None,
        categories: list[str] | None,
    ):
        if profile:
            low, high = profile.price_range
            stmt = stmt.where(ProductRecord.price >= low, ProductRecord.price <= high)

        explicit_categories = categories is not None
        selected_categories = self._clean_values(categories or [])
        if not selected_categories and profile:
            selected_categories = self._clean_values(profile.preferred_categories)

        if selected_categories:
            if explicit_categories:
                stmt = stmt.where(ProductRecord.category.in_(selected_categories))
            else:
                stmt = stmt.where(
                    or_(
                        ProductRecord.category.in_(selected_categories),
                        ProductRecord.hot_score > 0,
                    )
                )
        return stmt

    def _clean_values(self, values: Iterable[str]) -> list[str]:
        return [value for value in values if value]

    def _to_product(self, record: ProductRecord, stock: int | None) -> Product:
        try:
            tags = json.loads(record.tags_json or "[]")
        except json.JSONDecodeError:
            tags = []
        if not isinstance(tags, list):
            tags = []

        return Product(
            product_id=record.product_id,
            name=record.name,
            category=record.category,
            price=record.price,
            description=record.description,
            brand=record.brand,
            seller_id=record.seller_id,
            stock=max((stock or 0), 0),
            tags=[str(tag) for tag in tags],
            score=record.hot_score,
            image_url=record.image_url,
        )
