"""Source-grounded context tools for customer-service specialists."""

import json

from langchain_core.tools import tool

from rag.retrieval.rag_utils import retrieve_documents


def _compact(doc: dict) -> dict:
    return {
        "chunk_id": doc.get("chunk_id", ""),
        "filename": doc.get("filename", ""),
        "page_number": doc.get("page_number"),
        "section_title": doc.get("section_title", ""),
        "text": str(doc.get("text") or "")[:1000],
    }


def _context(query: str) -> dict:
    retrieved = retrieve_documents(query, top_k=6)
    return {
        "query": query,
        "context_chunks": [_compact(doc) for doc in retrieved.get("docs", [])],
        "retrieval_meta": retrieved.get("meta", {}),
    }


@tool("prepare_faq")
def prepare_faq(
    topic: str,
    category: str = "",
    brand: str = "",
    document_type: str = "",
    count: int = 5,
    tone: str = "professional",
) -> str:
    """Prepare knowledge-base evidence and constraints for a customer FAQ."""
    query = " ".join(
        part for part in [topic, category, brand, document_type, "常见问题"] if part
    )
    return json.dumps(
        {
            "task": "generate_faq",
            "requirements": {
                "topic": topic,
                "category": category,
                "brand": brand,
                "document_type": document_type,
                "count": max(1, min(int(count or 5), 20)),
                "tone": tone,
            },
            **_context(query),
        },
        ensure_ascii=False,
    )


@tool("prepare_sales_script")
def prepare_sales_script(
    customer_need: str,
    category: str = "",
    brand: str = "",
    channel: str = "online",
    tone: str = "professional",
) -> str:
    """Prepare evidence for a grounded sales and service conversation script."""
    query = " ".join(
        part
        for part in [customer_need, category, brand, "商品说明 售后 购买建议"]
        if part
    )
    return json.dumps(
        {
            "task": "generate_sales_script",
            "requirements": {
                "customer_need": customer_need,
                "category": category,
                "brand": brand,
                "channel": channel,
                "tone": tone,
            },
            **_context(query),
        },
        ensure_ascii=False,
    )


@tool("prepare_reply_review")
def prepare_reply_review(
    customer_message: str,
    agent_reply: str,
    policy_context: str = "",
    category: str = "",
    brand: str = "",
    max_score: int = 100,
) -> str:
    """Prepare policy evidence and inputs for customer-service reply review."""
    query = " ".join(
        part
        for part in [customer_message, category, brand, policy_context, "客服规则"]
        if part
    )
    return json.dumps(
        {
            "task": "review_service_reply",
            "inputs": {
                "customer_message": customer_message,
                "agent_reply": agent_reply,
                "policy_context": policy_context,
                "category": category,
                "brand": brand,
                "max_score": max(1, min(int(max_score or 100), 100)),
            },
            **_context(query),
        },
        ensure_ascii=False,
    )
