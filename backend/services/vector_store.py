from __future__ import annotations

from typing import Any

import structlog
from core.config import get_settings
from core.model_clients import embedding_collection_name

logger = structlog.get_logger()


class MilvusVectorStore:
    """Milvus-backed vector lookup for product and user embeddings."""

    def __init__(self):
        settings = get_settings()
        self.host = settings.milvus_host
        self.port = settings.milvus_port
        self.product_collection = embedding_collection_name(
            settings.milvus_product_collection or settings.milvus_collection
        )
        self.user_collection = embedding_collection_name(settings.milvus_user_collection)
        self.dimension = settings.embedding_dimension
        self._connected = False

    def _connect(self) -> bool:
        if self._connected:
            return True
        try:
            from pymilvus import connections

            connections.connect(
                alias="default",
                host=self.host,
                port=str(self.port),
                timeout=10,
            )
            self._connected = True
            return True
        except Exception as exc:
            logger.warning("milvus.connect_failed", error=str(exc))
            return False

    async def search_products(
        self,
        query_vector: list[float],
        limit: int,
    ) -> list[str]:
        if not query_vector or not self._connect():
            return []

        try:
            from pymilvus import Collection, utility

            if not utility.has_collection(self.product_collection):
                return []
            collection = Collection(self.product_collection)
            collection.load()
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=limit,
                output_fields=["product_id"],
            )
            return self._ids_from_hits(results, "product_id")
        except Exception as exc:
            logger.warning("milvus.product_search_failed", error=str(exc))
            return []

    async def upsert_product_embedding(
        self,
        product_id: str,
        embedding: list[float],
    ) -> bool:
        return await self._upsert_embedding(
            collection_name=self.product_collection,
            id_field="product_id",
            id_value=product_id,
            embedding=embedding,
        )

    async def upsert_product_embeddings(
        self,
        rows: list[tuple[str, list[float]]],
    ) -> bool:
        """Write one product batch and make it durable with a single flush."""
        return await self._upsert_embeddings(
            collection_name=self.product_collection,
            id_field="product_id",
            rows=rows,
        )

    async def upsert_user_embedding(
        self,
        user_id: str,
        embedding: list[float],
    ) -> bool:
        return await self._upsert_embedding(
            collection_name=self.user_collection,
            id_field="user_id",
            id_value=user_id,
            embedding=embedding,
        )

    async def get_user_embedding(self, user_id: str) -> list[float]:
        if not user_id or not self._connect():
            return []

        try:
            from pymilvus import Collection, utility

            if not utility.has_collection(self.user_collection):
                return []
            collection = Collection(self.user_collection)
            collection.load()
            rows = collection.query(
                expr=f'user_id == "{user_id}"',
                output_fields=["embedding"],
                limit=1,
            )
            if not rows:
                return []
            embedding = rows[0].get("embedding", [])
            return list(embedding) if embedding else []
        except Exception as exc:
            logger.warning("milvus.user_embedding_failed", error=str(exc))
            return []

    def reset_seed_collections(self) -> bool:
        """Drop seed-owned vector collections so regenerated data cannot mix with old vectors."""
        if not self._connect():
            return False

        ok = True
        for collection_name in {
            self.product_collection,
            self.user_collection,
        }:
            if not self.drop_collection(collection_name):
                ok = False
        return ok

    def drop_collection(self, collection_name: str) -> bool:
        if not collection_name or not self._connect():
            return False

        try:
            from pymilvus import utility

            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info("milvus.collection_dropped", collection=collection_name)
            return True
        except Exception as exc:
            logger.warning(
                "milvus.drop_collection_failed",
                collection=collection_name,
                error=str(exc),
            )
            return False

    def _ids_from_hits(self, results: Any, field_name: str) -> list[str]:
        ids: list[str] = []
        for hit in results[0] if results else []:
            value = None
            if hasattr(hit, "entity"):
                value = hit.entity.get(field_name)
            if value:
                ids.append(str(value))
        return ids

    async def _upsert_embedding(
        self,
        collection_name: str,
        id_field: str,
        id_value: str,
        embedding: list[float],
    ) -> bool:
        return await self._upsert_embeddings(
            collection_name=collection_name,
            id_field=id_field,
            rows=[(id_value, embedding)],
        )

    async def _upsert_embeddings(
        self,
        collection_name: str,
        id_field: str,
        rows: list[tuple[str, list[float]]],
    ) -> bool:
        if (
            not rows
            or any(not id_value or not embedding for id_value, embedding in rows)
            or not self._connect()
        ):
            return False

        try:
            collection = self._ensure_collection(collection_name, id_field)
            collection.upsert(
                [
                    [id_value for id_value, _ in rows],
                    [embedding for _, embedding in rows],
                ]
            )
            collection.flush()
            return True
        except Exception as exc:
            logger.warning(
                "milvus.upsert_failed",
                collection=collection_name,
                count=len(rows),
                error=str(exc),
            )
            return False

    def _ensure_collection(self, collection_name: str, id_field: str):
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            utility,
        )

        if utility.has_collection(collection_name):
            return Collection(collection_name)

        fields = [
            FieldSchema(
                name=id_field,
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=128,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dimension,
            ),
        ]
        collection = Collection(
            name=collection_name,
            schema=CollectionSchema(fields=fields),
        )
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )
        return collection
