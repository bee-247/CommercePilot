"""Sales chat and recommendation API routes."""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from core.application_state import (
    chat_router_agent,
    conversation_context_agent,
    conversation_memory,
    conversation_reply_agent,
    image_understanding_agent,
    metrics_collector,
    product_repository,
    supervisor,
)
from models.schemas import (
    AgentResult,
    ChatHistoryMessage,
    ChatRequest,
    ChatResponse,
    Product,
    RecommendationRequest,
    RecommendationResponse,
)
from services.token_counter import build_token_usage
from core.config import get_settings


logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["sales-chat"])


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """使用Supervisor编排器进行推荐 (生产推荐用法)"""
    response = await supervisor.recommend(request)
    _collect_metrics(response)
    return response


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Web chat entry: route conversational turns before recommending products."""
    memory = conversation_memory.get(request.user_id, request.session_id)
    history = conversation_memory.history_for_agent(memory)
    route_result = await chat_router_agent.run(
        message=request.message,
        history=history,
    )
    route_data = route_result.data
    if not route_data.get("needs_recommendation", True):
        return await _chat_without_recommendation(
            request=request,
            memory=memory,
            history=history,
            route_result=route_result,
        )

    available_categories = await product_repository.list_active_categories()
    context_result = await conversation_context_agent.run(
        message=request.message,
        history=history,
        available_categories=available_categories,
    )
    rewritten = context_result.data
    query = rewritten.get("standalone_query") or request.message
    categories = rewritten.get("categories") or []
    logger.info(
        "conversation_context.resolved",
        success=context_result.success,
        categories=categories,
        shopping_goal_count=len(rewritten.get("shopping_goals") or []),
    )
    long_term_memory = conversation_memory.long_term_candidates(
        user_id=request.user_id,
        limit=8,
        query=query,
    )

    context: dict[str, Any] = {
        "query": query,
        "original_query": request.message,
        "source": "web_chat",
        "conversation_history": [
            _history_item_to_dict(item) for item in history[-8:]
        ],
        "session_memory": memory.to_context(),
        "long_term_memory": long_term_memory,
        "conversation_context": rewritten,
    }
    if categories:
        context["categories"] = categories
    if rewritten.get("shopping_goals"):
        context["shopping_goals"] = rewritten.get("shopping_goals")
    if rewritten.get("budget"):
        context["budget"] = rewritten.get("budget")
    if rewritten.get("constraints"):
        context["constraints"] = rewritten.get("constraints")
    if rewritten.get("avoid_categories"):
        context["avoid_categories"] = rewritten.get("avoid_categories")
    if request.category:
        context["category"] = request.category
    rec_request = RecommendationRequest(
        user_id=request.user_id,
        scene="chat",
        num_items=request.num_items,
        context=context,
    )
    response = await supervisor.recommend(rec_request)
    response.agent_results["chat_router"] = route_result
    response.agent_results["conversation_context"] = context_result
    _collect_metrics(response)
    product_result = response.agent_results.get("product_rec")
    answer = _build_chat_answer(request.message, response)
    await conversation_memory.update(
        memory=memory,
        user_message=request.message,
        assistant_message=answer,
        context=context,
        response=response,
    )
    if route_result.data.get("needs_memory_update"):
        saved_memories = await conversation_memory.update_long_term_from_message(
            memory=memory,
            user_message=request.message,
        )
        route_result.data["updated_memories"] = saved_memories
    token_usage = build_token_usage(
        request_payload={
            "message": request.message,
            "history": [_history_item_to_dict(item) for item in history[-8:]],
            "session_memory": context["session_memory"],
            "context": context,
        },
        response_payload={"answer": answer},
        session_memory=memory.to_context(),
        agent_trace=_agent_trace_payload(response),
        compression_payload=_compression_payload(memory),
        summary=memory.summary,
        recent_messages=memory.messages[-8:],
        pending_turns=memory.turns,
    )
    return ChatResponse(
        answer=answer,
        recall_strategy=getattr(product_result, "recall_strategy", "")
        if product_result
        else "",
        recall_reason=product_result.data.get("recall_reason", "")
        if product_result
        else "",
        recommendation=response,
        token_usage=token_usage,
        rag_trace=response.rag_trace,
        citations=response.citations,
    )


async def _chat_without_recommendation(
    request: ChatRequest,
    memory: Any,
    history: list[ChatHistoryMessage],
    route_result: AgentResult,
) -> ChatResponse:
    context: dict[str, Any] = {
        "query": request.message,
        "original_query": request.message,
        "source": "web_chat",
        "conversation_history": [
            _history_item_to_dict(item) for item in history[-8:]
        ],
        "session_memory": memory.to_context(),
        "chat_route": route_result.data,
    }
    reply_result = await conversation_reply_agent.run(
        message=request.message,
        history=history,
        context=context,
    )
    answer = str(reply_result.data.get("answer") or "").strip()
    if not answer:
        answer = "好的，我明白了。"

    response = _empty_recommendation_response(
        user_id=request.user_id,
        agent_results={
            "chat_router": route_result,
            "conversation_reply": reply_result,
        },
    )
    _collect_metrics(response)
    await conversation_memory.update(
        memory=memory,
        user_message=request.message,
        assistant_message=answer,
        context=context,
        response=response,
    )
    if route_result.data.get("needs_memory_update"):
        saved_memories = await conversation_memory.update_long_term_from_message(
            memory=memory,
            user_message=request.message,
        )
        reply_result.data["updated_memories"] = saved_memories
    token_usage = build_token_usage(
        request_payload={
            "message": request.message,
            "history": [_history_item_to_dict(item) for item in history[-8:]],
            "session_memory": context["session_memory"],
            "context": context,
        },
        response_payload={"answer": answer},
        session_memory=memory.to_context(),
        agent_trace=_agent_trace_payload(response),
        compression_payload=_compression_payload(memory),
        summary=memory.summary,
        recent_messages=memory.messages[-8:],
        pending_turns=memory.turns,
    )
    return ChatResponse(
        answer=answer,
        recall_strategy="skipped",
        recall_reason=str(route_result.data.get("reason") or "no_recommendation_needed"),
        recommendation=response,
        token_usage=token_usage,
        rag_trace=response.rag_trace,
        citations=response.citations,
    )


@router.post("/chat/image", response_model=ChatResponse)
async def chat_with_image(
    message: str = Form(""),
    user_id: str = Form("web_user"),
    session_id: str | None = Form(None),
    history: str = Form("[]"),
    image: UploadFile = File(...),
):
    """Multimodal chat entry: understand an uploaded image before recommendation."""
    content_type = image.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持图片文件")
    max_bytes = get_settings().max_image_upload_mb * 1024 * 1024
    image_bytes = await image.read(max_bytes + 1)
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"图片不能超过 {get_settings().max_image_upload_mb} MB",
        )
    if not image_bytes:
        raise HTTPException(status_code=400, detail="图片内容为空")
    content_type = image.content_type or "image/jpeg"
    image_data_url = (
        f"data:{content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    )

    available_categories = await product_repository.list_active_categories()
    image_result = await image_understanding_agent.run(
        image_data_url=image_data_url,
        message=message,
        available_categories=available_categories,
    )
    query = image_result.data.get("query") or message or "根据图片推荐相似商品"
    category = image_result.data.get("category") or None
    memory = conversation_memory.get(user_id, session_id)
    memory_history = conversation_memory.history_for_agent(memory)
    context_result = await conversation_context_agent.run(
        message=message or query,
        history=memory_history,
        available_categories=available_categories,
        image_summary=image_result.data.get("summary", ""),
    )
    rewritten = context_result.data
    rewritten_query = rewritten.get("standalone_query") or query
    categories = rewritten.get("categories") or []
    logger.info(
        "conversation_context.resolved",
        success=context_result.success,
        categories=categories,
        shopping_goal_count=len(rewritten.get("shopping_goals") or []),
    )
    long_term_memory = conversation_memory.long_term_candidates(
        user_id=user_id,
        limit=8,
        query=rewritten_query,
    )

    context: dict[str, Any] = {
        "query": rewritten_query,
        "image_query": query,
        "original_query": message,
        "source": "web_image_chat",
        "skip_category_filter": True,
        "conversation_history": [
            _history_item_to_dict(item) for item in memory_history[-8:]
        ],
        "session_memory": memory.to_context(),
        "long_term_memory": long_term_memory,
        "conversation_context": rewritten,
        "image_summary": image_result.data.get("summary", ""),
        "image_attributes": image_result.data.get("attributes", []),
        "image_category": category or "",
    }
    if categories:
        context["categories"] = categories
    elif category:
        context["category"] = category
    if rewritten.get("shopping_goals"):
        context["shopping_goals"] = rewritten.get("shopping_goals")
    if rewritten.get("budget"):
        context["budget"] = rewritten.get("budget")
    if rewritten.get("constraints"):
        context["constraints"] = rewritten.get("constraints")
    rec_request = RecommendationRequest(
        user_id=user_id,
        scene="image_chat",
        num_items=6,
        context=context,
    )
    response = await supervisor.recommend(rec_request)
    response.agent_results["image_understanding"] = image_result
    response.agent_results["conversation_context"] = context_result
    _collect_metrics(response)

    product_result = response.agent_results.get("product_rec")
    answer = _build_image_chat_answer(message, image_result.data, response)
    await conversation_memory.update(
        memory=memory,
        user_message=message or query,
        assistant_message=answer,
        context=context,
        response=response,
    )
    token_usage = build_token_usage(
        request_payload={
            "message": message or query,
            "history": [_history_item_to_dict(item) for item in memory_history[-8:]],
            "session_memory": context["session_memory"],
            "image_summary": image_result.data.get("summary", ""),
            "image_attributes": image_result.data.get("attributes", []),
            "context": context,
        },
        response_payload={"answer": answer},
        session_memory=memory.to_context(),
        agent_trace=_agent_trace_payload(response),
        compression_payload=_compression_payload(memory),
        summary=memory.summary,
        recent_messages=memory.messages[-8:],
        pending_turns=memory.turns,
    )
    return ChatResponse(
        answer=answer,
        recall_strategy=getattr(product_result, "recall_strategy", "")
        if product_result
        else "",
        recall_reason=product_result.data.get("recall_reason", "")
        if product_result
        else "",
        recommendation=response,
        token_usage=token_usage,
        rag_trace=response.rag_trace,
        citations=response.citations,
    )


@router.post("/chat/session/end")
async def end_chat_session(
    user_id: str = "web_user",
    session_id: str | None = None,
    clear: bool = True,
):
    """Summarize a finished session and extract long-term memory candidates."""
    return await conversation_memory.end_session(
        user_id=user_id,
        session_id=session_id,
        clear=clear,
    )


def _collect_metrics(response: RecommendationResponse):
    for name, result in response.agent_results.items():
        metrics_collector.record_agent_call(
            agent_name=name,
            success=result.success,
            latency_ms=result.latency_ms,
        )


def _empty_recommendation_response(
    user_id: str,
    agent_results: dict[str, AgentResult],
) -> RecommendationResponse:
    return RecommendationResponse(
        request_id=str(uuid.uuid4()),
        user_id=user_id,
        products=[],
        marketing_copies=[],
        experiment_group="none",
        agent_results=agent_results,
        total_latency_ms=0.0,
    )


def _parse_history(raw: str) -> list[ChatHistoryMessage]:
    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    history = []
    for item in data[-8:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = str(item.get("content") or "").strip()
        if role in {"user", "agent", "assistant"} and content:
            history.append(ChatHistoryMessage(role=role, content=content))
    return history


def _history_item_to_dict(item: Any) -> dict[str, str]:
    if isinstance(item, ChatHistoryMessage):
        return item.model_dump()
    if isinstance(item, dict):
        return {
            "role": str(item.get("role") or ""),
            "content": str(item.get("content") or ""),
        }
    return {"role": "", "content": ""}


def _agent_trace_payload(response: RecommendationResponse) -> dict[str, Any]:
    return {
        "products": [
            {
                "product_id": product.product_id,
                "name": product.name,
                "category": product.category,
                "price": product.price,
                "brand": product.brand,
                "tags": product.tags,
                "description": product.description,
            }
            for product in response.products
        ],
        "marketing_copies": response.marketing_copies,
        "agent_results": {
            name: {
                "success": result.success,
                "confidence": result.confidence,
                "error": result.error,
                "data": result.data,
            }
            for name, result in response.agent_results.items()
        },
    }


def _compression_payload(memory: Any) -> dict[str, Any]:
    return {
        "previous_summary": memory.summary,
        "recent_turns": memory.turns,
        "current_memory": memory.to_context(),
    }


def _build_chat_answer(message: str, response: RecommendationResponse) -> str:
    count = len(response.products)
    guide_result = response.agent_results.get("shopping_guide")
    if guide_result and guide_result.success and guide_result.data is not None:
        answer = getattr(guide_result, "answer", "")
        if answer:
            return _append_product_lines(answer, response)

    if count == 0:
        return (
            "我已经理解你的需求，但当前商品库或库存里没有可推荐的商品。"
            "可以先导入商品、库存并建立向量索引后再试。"
        )

    lines = [f"我根据你的需求「{message}」找到了这些推荐："]
    lines.extend(_format_product_lines(response))
    return "\n".join(lines)


def _build_image_chat_answer(
    message: str,
    image_data: dict[str, Any],
    response: RecommendationResponse,
) -> str:
    count = len(response.products)
    summary = image_data.get("summary") or "图片内容"
    guide_result = response.agent_results.get("shopping_guide")
    if guide_result and guide_result.success:
        answer = getattr(guide_result, "answer", "")
        if answer:
            return _append_product_lines(f"我看到了{summary}。\n{answer}", response)

    if count == 0:
        return (
            f"我看到了{summary}，并尝试根据图片特征进行向量召回，"
            "但当前商品库、库存或向量索引里没有匹配结果。"
        )

    user_text = f"结合你的补充「{message}」，" if message else ""
    lines = [
        f"我看到了{summary}，{user_text}根据图片特征找到了这些推荐："
    ]
    lines.extend(_format_product_lines(response))
    return "\n".join(lines)


def _format_product_lines(
    response: RecommendationResponse,
    products: list[Product] | None = None,
) -> list[str]:
    copy_by_product = {
        item.get("product_id"): item.get("copy", "")
        for item in response.marketing_copies
        if item.get("product_id")
    }
    lines = []
    for index, product in enumerate(products or response.products, start=1):
        line = f"{index}. {product.name}，¥{product.price:.2f}"
        if product.category:
            line += f"，{product.category}"
        copy = copy_by_product.get(product.product_id)
        if copy:
            line += f"\n   {copy}"
        lines.append(line)
    return lines


def _append_product_lines(answer: str, response: RecommendationResponse) -> str:
    if not response.products:
        return answer
    products = _products_used_by_guide(response)
    if not products:
        return answer
    product_lines = _format_product_lines(response, products)
    mentioned_names = [
        product.name for product in products if product.name and product.name in answer
    ]
    if len(mentioned_names) == len(products):
        return answer
    return "\n".join([answer.rstrip(), "", "具体推荐：", *product_lines])


def _products_used_by_guide(response: RecommendationResponse) -> list[Product]:
    guide_result = response.agent_results.get("shopping_guide")
    if guide_result and guide_result.data is not None:
        displayed_ids = guide_result.data.get("displayed_product_ids")
        if isinstance(displayed_ids, list):
            allowed = {str(product_id) for product_id in displayed_ids}
            return [product for product in response.products if product.product_id in allowed]
    return response.products
