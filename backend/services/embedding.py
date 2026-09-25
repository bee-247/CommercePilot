"""One dense embedding model for products, users, documents and queries."""

from __future__ import annotations

import math

import structlog
from core.config import get_settings
from openai import AsyncOpenAI, OpenAI

logger = structlog.get_logger()


class EmbeddingService:
    def __init__(self):
        settings = get_settings()
        self.embedding_dimension = settings.embedding_dimension
        self.embedding_model = settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        self.api_key = settings.embedding_api_key
        self.base_url = settings.embedding_base_url
        self._async_client: AsyncOpenAI | None = None
        self._sync_client: OpenAI | None = None

    def _connection_parameters(self) -> dict:
        if not self.api_key or not self.base_url or not self.embedding_model:
            raise ValueError(
                "请配置 EMBEDDING_MODEL、EMBEDDING_API_KEY 和 EMBEDDING_BASE_URL"
            )
        return {"api_key": self.api_key, "base_url": self.base_url, "timeout": 60.0}

    def _request(self, texts: list[str]) -> dict:
        return {
            "model": self.embedding_model,
            "input": texts,
            "dimensions": self.embedding_dimension,
            "encoding_format": "float",
        }

    def _vectors(self, response, count: int) -> list[list[float]]:
        rows = list(response.data)
        if len(rows) != count:
            raise ValueError("Embedding 返回数量或索引与输入不一致")
        expected_indices = list(range(count))
        returned_indices = [row.index for row in rows]
        if sorted(returned_indices) == expected_indices:
            rows.sort(key=lambda item: item.index)
        elif not (
            self.embedding_model == "qwen3.7-text-embedding-flash"
            and count > 1
            and returned_indices == [0] * count
        ):
            raise ValueError("Embedding 返回数量或索引与输入不一致")
        else:
            # This model currently returns index=0 for every item while preserving
            # the input order. Limit the workaround to that exact provider defect.
            logger.debug(
                "embedding.duplicate_indices_using_response_order",
                model=self.embedding_model,
                count=count,
            )
        vectors = []
        for row in rows:
            vector = [float(value) for value in row.embedding]
            if len(vector) != self.embedding_dimension:
                raise ValueError("Embedding 维度与 EMBEDDING_DIMENSION 不一致")
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Embedding 包含非有限数值")
            norm = math.hypot(*vector)
            if not math.isfinite(norm) or not norm:
                raise ValueError("Embedding 向量范数无效")
            # RAG uses inner product, sales uses cosine.
            vectors.append([value / norm for value in vector])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Synchronous adapter for document ingestion and RAG retrieval."""
        if not texts:
            return []
        if self._sync_client is None:
            self._sync_client = OpenAI(**self._connection_parameters())
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._sync_client.embeddings.create(**self._request(batch))
            vectors.extend(self._vectors(response, len(batch)))
        return vectors

    async def embed_documents_async(self, texts: list[str]) -> list[list[float]]:
        """Asynchronous batch adapter for concurrent indexing jobs."""
        if not texts:
            return []
        if self._async_client is None:
            self._async_client = AsyncOpenAI(**self._connection_parameters())
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = await self._async_client.embeddings.create(
                **self._request(batch)
            )
            vectors.extend(self._vectors(response, len(batch)))
        return vectors

    async def embed_text(self, text: str) -> list[float]:
        """Async adapter preserving sales recall fallback on provider failure."""
        text = str(text or "").strip()
        if not text:
            return []
        try:
            return (await self.embed_documents_async([text]))[0]
        except Exception as exc:
            logger.warning("embedding.failed", error=str(exc))
            return []
