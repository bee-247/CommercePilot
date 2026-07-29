"""Service boundary for sales recommendation augmented by RAG evidence."""

from __future__ import annotations

import asyncio
from typing import Any

from core.config import get_settings
from rag.retrieval.rag_utils import retrieve_documents

from .context_builder import FusionContextBuilder


class SalesRagService:
    """Augment sales recommendation context with public RAG evidence."""

    def __init__(self, context_builder: FusionContextBuilder | None = None) -> None:
        self.context_builder = context_builder or FusionContextBuilder()
        settings = get_settings()
        self.enabled = settings.rag_augmentation_enabled
        self.top_k = max(1, min(settings.rag_augmentation_top_k, 8))

    async def build_context(
        self,
        *,
        user_message: str,
        sales_context: dict[str, Any] | None = None,
        rag_context: list[dict[str, Any]] | None = None,
        products: list[dict[str, Any]] | None = None,
        memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.context_builder.build(
            user_message=user_message,
            sales_context=sales_context,
            rag_context=rag_context,
            products=products,
            memory=memory,
        )

    async def augment(
        self,
        *,
        user_message: str,
        sales_context: dict[str, Any],
        products: list[dict[str, Any]] | None = None,
        memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a copied sales context enriched with evidence and trace.

        Retrieval failures deliberately degrade to an empty evidence set so the
        sales recommendation flow remains usable without the knowledge store.
        """
        context = dict(sales_context)
        if not self.enabled or not user_message.strip():
            context["rag_context"] = []
            context["rag_trace"] = {
                "enabled": self.enabled,
                "applied": False,
                "reason": "disabled_or_empty_query",
            }
            context["citations"] = []
            return context

        try:
            retrieved = await asyncio.to_thread(
                retrieve_documents,
                user_message.strip(),
                self.top_k,
            )
            documents = retrieved.get("docs", [])
            metadata = retrieved.get("meta", {})
            citations = self._citations(documents)
            context["rag_context"] = documents
            context["citations"] = citations
            context["rag_trace"] = {
                "enabled": True,
                "applied": bool(documents),
                "query": user_message.strip(),
                "retrieval_mode": metadata.get("retrieval_mode"),
                "rerank_applied": metadata.get("rerank_applied"),
                "auto_merge_applied": metadata.get("auto_merge_applied"),
                "candidate_k": metadata.get("candidate_k"),
                "retrieved_chunks": documents,
                "error": metadata.get("rerank_error"),
            }
        except Exception as exc:
            context["rag_context"] = []
            context["citations"] = []
            context["rag_trace"] = {
                "enabled": True,
                "applied": False,
                "query": user_message.strip(),
                "reason": "retrieval_unavailable",
                "error": str(exc),
            }

        context["fusion_context"] = await self.build_context(
            user_message=user_message,
            sales_context=sales_context,
            rag_context=context["rag_context"],
            products=products,
            memory=memory,
        )
        return context

    @staticmethod
    def _citations(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        citations = []
        seen: set[tuple[str, str, str]] = set()
        for document in documents:
            filename = str(document.get("filename") or "")
            page = str(document.get("page_number") or "")
            chunk_id = str(document.get("chunk_id") or "")
            key = (filename, page, chunk_id)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                {
                    "filename": filename,
                    "page_number": document.get("page_number"),
                    "section_title": document.get("section_title", ""),
                    "chunk_id": chunk_id,
                    "text": str(document.get("text") or "")[:500],
                }
            )
        return citations
