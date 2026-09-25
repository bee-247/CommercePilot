"""Service boundary for sales recommendation augmented by RAG evidence."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from core.config import get_settings
from rag.retrieval.rag_utils import retrieve_documents, retrieve_documents_fast

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
        return await self._augment_with(
            retriever=retrieve_documents,
            retrieval_profile="standard",
            user_message=user_message,
            sales_context=sales_context,
            products=products,
            memory=memory,
        )

    async def augment_fast(
        self,
        *,
        user_message: str,
        sales_context: dict[str, Any],
        products: list[dict[str, Any]] | None = None,
        memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Enrich context through the low-latency retrieval profile."""
        return await self._augment_with(
            retriever=retrieve_documents_fast,
            retrieval_profile="fast",
            user_message=user_message,
            sales_context=sales_context,
            products=products,
            memory=memory,
        )

    async def _augment_with(
        self,
        *,
        retriever: Callable[..., dict[str, Any]],
        retrieval_profile: str,
        user_message: str,
        sales_context: dict[str, Any],
        products: list[dict[str, Any]] | None,
        memory: dict[str, Any] | None,
    ) -> dict[str, Any]:
        context = dict(sales_context)
        if not self.enabled or not user_message.strip():
            context["rag_context"] = []
            context["rag_trace"] = {
                "enabled": self.enabled,
                "applied": False,
                "reason": "disabled_or_empty_query",
                "retrieval_profile": retrieval_profile,
            }
            context["citations"] = []
            return context

        try:
            if products:
                documents, metadata = await self._retrieve_product_evidence(
                    retriever=retriever,
                    user_message=user_message.strip(),
                    products=products,
                )
            else:
                retrieved = await asyncio.to_thread(
                    retriever,
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
                "retrieval_profile": metadata.get(
                    "retrieval_profile",
                    retrieval_profile,
                ),
                "retrieval_mode": metadata.get("retrieval_mode"),
                "retrieval_modes": metadata.get("retrieval_modes") or [],
                "product_query_count": metadata.get("product_query_count", 0),
                "parallel_retrieval": bool(
                    metadata.get("parallel_retrieval")
                ),
                "evidence_queries": metadata.get("evidence_queries") or [],
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
                "retrieval_profile": retrieval_profile,
            }

        context["fusion_context"] = await self.build_context(
            user_message=user_message,
            sales_context=sales_context,
            rag_context=context["rag_context"],
            products=products,
            memory=memory,
        )
        return context

    async def _retrieve_product_evidence(
        self,
        *,
        retriever: Callable[..., dict[str, Any]],
        user_message: str,
        products: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        targets = [
            product
            for product in products[:12]
            if str(product.get("product_id") or product.get("name") or "").strip()
        ]
        per_product_top_k = max(1, min(self.top_k, 2))
        semaphore = asyncio.Semaphore(8)

        async def retrieve_one(product: dict[str, Any]) -> dict[str, Any]:
            query = self._product_evidence_query(user_message, product)
            async with semaphore:
                try:
                    result = await asyncio.to_thread(
                        retriever,
                        query,
                        per_product_top_k,
                    )
                except Exception as exc:
                    result = {
                        "docs": [],
                        "meta": {"retrieval_mode": "failed", "error": str(exc)},
                    }
            return {
                "product": product,
                "query": query,
                "result": result,
            }

        retrieved = await asyncio.gather(*(retrieve_one(item) for item in targets))
        documents: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        modes: list[str] = []
        errors: list[str] = []
        rerank_applied = False
        auto_merge_applied = False

        for item in retrieved:
            product = item["product"]
            result = item["result"]
            metadata = result.get("meta") or {}
            mode = str(metadata.get("retrieval_mode") or "unknown")
            if mode not in modes:
                modes.append(mode)
            rerank_applied = rerank_applied or bool(
                metadata.get("rerank_applied")
            )
            auto_merge_applied = auto_merge_applied or bool(
                metadata.get("auto_merge_applied")
            )
            error = metadata.get("rerank_error") or metadata.get("error")
            if error:
                errors.append(str(error))

            product_id = str(product.get("product_id") or "")
            product_name = str(product.get("name") or "")
            for document in result.get("docs") or []:
                chunk_key = str(
                    document.get("chunk_id")
                    or document.get("id")
                    or document.get("text")
                    or ""
                )
                key = (product_id or product_name, chunk_key)
                if key in seen:
                    continue
                seen.add(key)
                documents.append(
                    {
                        **document,
                        "target_product_id": product_id,
                        "target_product_name": product_name,
                        "evidence_query": item["query"],
                    }
                )

        return documents, {
            "retrieval_mode": "product_evidence_parallel",
            "retrieval_modes": modes,
            "product_query_count": len(targets),
            "parallel_retrieval": len(targets) > 1,
            "candidate_k": per_product_top_k,
            "candidate_count": len(documents),
            "rerank_applied": rerank_applied,
            "auto_merge_applied": auto_merge_applied,
            "rerank_error": "; ".join(dict.fromkeys(errors)) if errors else None,
            "evidence_queries": [
                {
                    "product_id": str(item["product"].get("product_id") or ""),
                    "product_name": str(item["product"].get("name") or ""),
                    "query": item["query"],
                }
                for item in retrieved
            ],
        }

    @staticmethod
    def _product_evidence_query(
        user_message: str,
        product: dict[str, Any],
    ) -> str:
        prefix = "；".join(
            part
            for part in [
                f"商品ID：{str(product.get('product_id') or '').strip()}",
                f"商品名称：{str(product.get('name') or '').strip()}",
                f"品牌：{str(product.get('brand') or '').strip()}",
                f"品类：{str(product.get('category') or '').strip()}",
            ]
            if not part.endswith("：")
        )
        return f"[候选商品] {prefix}\n[用户需求] {user_message}"

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
                    "target_product_id": document.get("target_product_id", ""),
                    "target_product_name": document.get(
                        "target_product_name",
                        "",
                    ),
                }
            )
        return citations
