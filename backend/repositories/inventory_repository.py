from __future__ import annotations

from sqlalchemy import select

from database.models import InventoryRecord
from database.session import ProductSessionLocal


class InventoryRepository:
    """Database-backed inventory lookup repository."""

    async def get_stock_map(self, product_ids: list[str]) -> dict[str, int]:
        if not product_ids:
            return {}

        with ProductSessionLocal() as session:
            rows = session.execute(
                select(
                    InventoryRecord.product_id,
                    InventoryRecord.stock,
                    InventoryRecord.reserved_stock,
                ).where(InventoryRecord.product_id.in_(product_ids))
            ).all()

        stock_map = {}
        for product_id, stock, reserved_stock in rows:
            stock_map[product_id] = max((stock or 0) - (reserved_stock or 0), 0)
        return stock_map
