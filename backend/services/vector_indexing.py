from __future__ import annotations

from models.schemas import Product
from repositories import ProductRepository, UserRepository
from services.embedding import EmbeddingService
from services.vector_store import MilvusVectorStore


class VectorIndexingService:
    """Builds product and user embeddings and writes them to Milvus."""

    def __init__(self):
        self.product_repository = ProductRepository()
        self.user_repository = UserRepository()
        self.embedding_service = EmbeddingService()
        self.vector_store = MilvusVectorStore()

    async def index_product(self, product: Product) -> bool:
        text = self._product_text(product)
        embedding = await self.embedding_service.embed_text(text)
        return await self.vector_store.upsert_product_embedding(
            product.product_id,
            embedding,
        )

    async def index_products(self, limit: int = 1000) -> dict[str, int]:
        products = await self.product_repository.list_products_for_index(limit=limit)
        indexed = 0
        for product in products:
            if await self.index_product(product):
                indexed += 1
        return {"total": len(products), "indexed": indexed}

    async def index_user(self, user_id: str, history_limit: int = 50) -> bool:
        product_ids = await self.user_repository.get_recent_product_ids(
            user_id=user_id,
            limit=history_limit,
        )
        products = await self.product_repository.get_products_by_ids(product_ids)
        text = self.user_repository.build_interaction_text(products)
        embedding = await self.embedding_service.embed_text(text)
        return await self.vector_store.upsert_user_embedding(user_id, embedding)

    def _product_text(self, product: Product) -> str:
        return " ".join(
            [
                product.name,
                product.description,
                product.category,
                product.brand,
                " ".join(product.tags),
            ]
        ).strip()
