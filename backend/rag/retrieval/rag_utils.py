from collections import defaultdict
from typing import List, Tuple, Dict, Any
import os

from ..storage.milvus_client import MilvusManager
from .embedding import embedding_service as _embedding_service
from ..storage.parent_chunk_store import ParentChunkStore
from core.config import get_settings
from core.model_clients import create_chat_model, rerank_endpoint
from services.reranking import get_reranking_service
from ..storage.database import SessionLocal
from customer_service.tool_context import get_service_username
from ..storage.models import Resource, User
from ..utils.token_usage_tracker import record_active_session_token_usage_from_message
from core.env import load_project_env

load_project_env()

_settings = get_settings()
RERANK_MODEL = _settings.rerank_model
AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))

# 全局初始化检索依赖（与 api 共用 embedding_service，保证 BM25 状态一致）
_milvus_manager = MilvusManager()
_parent_chunk_store = ParentChunkStore()

_stepback_model = None


def _get_rerank_endpoint() -> str:
    return "local" if _settings.rerank_model_path else rerank_endpoint()


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue
        parent_doc = dict(parent_map[parent_id])
        score = doc.get("score")
        if score is not None:
            parent_doc["score"] = max(float(parent_doc.get("score", score)), float(score))
        rerank_scores = [
            float(child["rerank_score"])
            for child in groups[parent_id]
            if child.get("rerank_score") is not None
        ]
        if rerank_scores:
            parent_doc["rerank_score"] = max(rerank_scores)
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        merged_docs.append(parent_doc)
        merged_count += 1

    deduped: List[dict] = []
    seen = set()
    for item in merged_docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merged_count


def _auto_merge_documents(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    if not AUTO_MERGE_ENABLED or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    # 两段自动合并：L3->L2，再 L2->L1。
    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    merged_docs.sort(
        key=lambda item: item.get("rerank_score", item.get("score", 0.0)),
        reverse=True,
    )
    merged_docs = merged_docs[:top_k]

    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    return merged_docs, {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
    }


def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    return get_reranking_service().rerank(query, docs, top_k)


def _get_stepback_model():
    global _stepback_model
    if not _settings.text_api_key or not _settings.text_llm:
        return None
    if _stepback_model is None:
        _stepback_model = create_chat_model(
            temperature=0.2,
        )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请将用户的具体问题抽象成更高层次、更概括的‘退步问题’，"
        "用于探寻背后的通用原理或核心概念。只输出退步问题一句话，不要解释。\n"
        f"用户问题：{query}"
    )
    try:
        response = model.invoke(prompt)
        record_active_session_token_usage_from_message(response)
        return (response.content or "").strip()
    except Exception:
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = (
        "请简要回答以下退步问题，提供通用原理/背景知识，"
        "控制在120字以内。只输出答案，不要列出推理过程。\n"
        f"退步问题：{step_back_question}"
    )
    try:
        response = model.invoke(prompt)
        record_active_session_token_usage_from_message(response)
        return (response.content or "").strip()
    except Exception:
        return ""


def generate_hypothetical_document(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请基于用户问题生成一段‘假设性文档’，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        response = model.invoke(prompt)
        record_active_session_token_usage_from_message(response)
        return (response.content or "").strip()
    except Exception:
        return ""


def step_back_expand(query: str) -> dict:
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    else:
        expanded_query = query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


def _escape_filter_value(value: str) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


def _build_retrieval_filter(
    *,
    category: str = "",
    brand: str = "",
    business_line: str = "",
    document_type: str = "",
    section_title: str = "",
    username: str = "",
) -> str:
    clauses = [f"chunk_level == {LEAF_RETRIEVE_LEVEL}"]
    scalar_filters = {
        "category": category,
        "brand": brand,
        "business_line": business_line,
        "document_type": document_type,
        "section_title": section_title,
    }
    for field, raw_value in scalar_filters.items():
        value = str(raw_value or "").strip()
        if value:
            clauses.append(f'{field} == "{_escape_filter_value(value)}"')
    allowed_ids = _allowed_resource_ids(username)
    if allowed_ids is not None:
        if not allowed_ids:
            clauses.append("resource_id in [-1]")
        else:
            clauses.append(f"resource_id in [{', '.join(str(item) for item in allowed_ids)}]")
    return " and ".join(clauses)


def _allowed_resource_ids(username: str = "") -> list[int] | None:
    """Return public + own resource ids. None means no DB scope could be applied."""
    db = SessionLocal()
    try:
        query = db.query(Resource.id).filter(Resource.status == "processed")
        user = None
        if username:
            user = db.query(User).filter(User.username == username).first()
        if user and user.role == "admin":
            return [row[0] for row in query.all()]
        if user:
            query = query.filter((Resource.visibility == "public") | (Resource.owner_id == user.id))
        else:
            query = query.filter(Resource.visibility == "public")
        return [row[0] for row in query.all()]
    except Exception:
        return None
    finally:
        db.close()


def retrieve_documents(
    query: str,
    top_k: int = 5,
    *,
    category: str = "",
    brand: str = "",
    business_line: str = "",
    document_type: str = "",
    section_title: str = "",
    username: str = "",
) -> Dict[str, Any]:
    candidate_k = max(top_k * 3, top_k)
    filter_expr = _build_retrieval_filter(
        category=category,
        brand=brand,
        business_line=business_line,
        document_type=document_type,
        section_title=section_title,
        username=username or get_service_username(),
    )
    try:
        dense_embeddings = _embedding_service.get_embeddings([query])
        dense_embedding = dense_embeddings[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query)

        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            filter_expr=filter_expr,
        )
        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
        merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
        rerank_meta["retrieval_mode"] = "hybrid"
        rerank_meta["candidate_k"] = candidate_k
        rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
        rerank_meta["filter_expr"] = filter_expr
        rerank_meta.update(merge_meta)
        return {"docs": merged_docs, "meta": rerank_meta}
    except Exception:
        try:
            dense_embeddings = _embedding_service.get_embeddings([query])
            dense_embedding = dense_embeddings[0]
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                filter_expr=filter_expr,
            )
            reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
            merged_docs, merge_meta = _auto_merge_documents(docs=reranked, top_k=top_k)
            rerank_meta["retrieval_mode"] = "dense_fallback"
            rerank_meta["candidate_k"] = candidate_k
            rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
            rerank_meta["filter_expr"] = filter_expr
            rerank_meta.update(merge_meta)
            return {"docs": merged_docs, "meta": rerank_meta}
        except Exception:
            return {
                "docs": [],
                "meta": {
                    "rerank_enabled": get_reranking_service().enabled,
                    "rerank_applied": False,
                    "rerank_model": RERANK_MODEL,
                    "rerank_endpoint": _get_rerank_endpoint(),
                    "rerank_error": "retrieve_failed",
                    "retrieval_mode": "failed",
                    "candidate_k": candidate_k,
                    "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                    "auto_merge_enabled": AUTO_MERGE_ENABLED,
                    "auto_merge_applied": False,
                    "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
                    "auto_merge_replaced_chunks": 0,
                    "auto_merge_steps": 0,
                    "candidate_count": 0,
                },
            }


def retrieve_documents_fast(
    query: str,
    top_k: int = 5,
    *,
    category: str = "",
    brand: str = "",
    business_line: str = "",
    document_type: str = "",
    section_title: str = "",
    username: str = "",
) -> Dict[str, Any]:
    """Run bounded hybrid retrieval without external rerank or parent merging."""
    result_limit = max(1, top_k)
    filter_expr = _build_retrieval_filter(
        category=category,
        brand=brand,
        business_line=business_line,
        document_type=document_type,
        section_title=section_title,
        username=username or get_service_username(),
    )
    try:
        dense_embedding = _embedding_service.get_embeddings([query])[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query)
        documents = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=result_limit,
            filter_expr=filter_expr,
        )
        mode = "hybrid_fast"
    except Exception:
        try:
            dense_embedding = _embedding_service.get_embeddings([query])[0]
            documents = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=result_limit,
                filter_expr=filter_expr,
            )
            mode = "dense_fast_fallback"
        except Exception:
            documents = []
            mode = "failed"

    ranked_documents = [
        {**document, "rrf_rank": index}
        for index, document in enumerate(documents[:result_limit], 1)
    ]
    return {
        "docs": ranked_documents,
        "meta": {
            "retrieval_mode": mode,
            "retrieval_profile": "fast",
            "candidate_k": result_limit,
            "candidate_count": len(ranked_documents),
            "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
            "filter_expr": filter_expr,
            "rerank_enabled": False,
            "rerank_applied": False,
            "rerank_model": None,
            "rerank_endpoint": "",
            "rerank_error": "retrieve_failed" if mode == "failed" else None,
            "auto_merge_enabled": False,
            "auto_merge_applied": False,
            "auto_merge_threshold": None,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        },
    }
