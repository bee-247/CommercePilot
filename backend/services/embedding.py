from __future__ import annotations

import structlog
from openai import AsyncOpenAI

from core.config import get_settings

logger = structlog.get_logger()


class EmbeddingService:
    """Thin wrapper around the configured embedding model."""

    def __init__(self):
        settings = get_settings()
        self.embedding_dimension = settings.embedding_dimension
        self.embedding_model = settings.embedding_model
        self._client = AsyncOpenAI(
            api_key=settings.embedding_api_key or settings.llm_api_key,
            base_url=settings.embedding_base_url or settings.llm_base_url,
        )

    async def embed_text(self, text: str) -> list[float]:
        text = str(text or "").strip()
        if not text:
            return []
        try:
            response = await self._client.embeddings.create(
                model=self.embedding_model,
                input=text,
                dimensions=self.embedding_dimension,
                encoding_format="float",
            )
            return list(response.data[0].embedding)
        except Exception as exc:
            logger.warning("embedding.failed", error=str(exc))
            return []
