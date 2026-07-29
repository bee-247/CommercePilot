from __future__ import annotations

from sqlalchemy import desc, select

from database.models import UserBehaviorRecord
from database.session import UserSessionLocal
from models.schemas import Product


class UserRepository:
    """Reads user profile and interaction history from the user database."""

    async def get_recent_behaviors(
        self, user_id: str, limit: int = 50
    ) -> list[dict[str, object]]:
        with UserSessionLocal() as session:
            rows = session.execute(
                select(
                    UserBehaviorRecord.product_id,
                    UserBehaviorRecord.behavior_type,
                    UserBehaviorRecord.weight,
                )
                .where(UserBehaviorRecord.user_id == user_id)
                .order_by(desc(UserBehaviorRecord.created_at))
                .limit(limit)
            ).all()
        return [
            {
                "product_id": product_id,
                "behavior_type": behavior_type,
                "weight": weight,
            }
            for product_id, behavior_type, weight in rows
        ]

    async def get_recent_product_ids(self, user_id: str, limit: int = 50) -> list[str]:
        with UserSessionLocal() as session:
            rows = session.execute(
                select(UserBehaviorRecord.product_id)
                .where(UserBehaviorRecord.user_id == user_id)
                .order_by(desc(UserBehaviorRecord.created_at))
                .limit(limit)
            ).all()
        return [product_id for (product_id,) in rows]

    async def list_behavior_sequences(
        self, limit: int = 5000
    ) -> dict[str, list[tuple[str, float]]]:
        with UserSessionLocal() as session:
            rows = session.execute(
                select(
                    UserBehaviorRecord.user_id,
                    UserBehaviorRecord.product_id,
                    UserBehaviorRecord.weight,
                )
                .order_by(desc(UserBehaviorRecord.created_at))
                .limit(limit)
            ).all()

        sequences: dict[str, list[tuple[str, float]]] = {}
        for user_id, product_id, weight in rows:
            sequences.setdefault(user_id, []).append((product_id, float(weight or 1.0)))
        return sequences

    def build_interaction_text(self, products: list[Product]) -> str:
        parts = []
        for product in products:
            text = " ".join(
                [
                    product.name,
                    product.category,
                    product.brand,
                    product.description,
                    " ".join(product.tags),
                ]
            )
            parts.append(text.strip())
        return "\n".join(part for part in parts if part)
