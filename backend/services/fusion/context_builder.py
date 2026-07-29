"""Build merged context for sales recommendation plus RAG augmentation."""

from __future__ import annotations

from typing import Any


class FusionContextBuilder:
    """Normalize sales, memory, product, and RAG snippets into one payload."""

    def build(
        self,
        *,
        user_message: str,
        sales_context: dict[str, Any] | None = None,
        rag_context: list[dict[str, Any]] | None = None,
        products: list[dict[str, Any]] | None = None,
        memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "user_message": user_message,
            "sales_context": sales_context or {},
            "rag_context": rag_context or [],
            "products": products or [],
            "memory": memory or {},
        }
