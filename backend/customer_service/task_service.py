"""Structured generation and verification for customer-service tools."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from customer_service.generation_tools import (
    prepare_faq,
    prepare_reply_review,
    prepare_sales_script,
)
from customer_service.output_schemas import (
    FaqSetOutput,
    ReplyReviewOutput,
    SalesScriptOutput,
)
from customer_service.tool_context import (
    reset_service_username,
    set_service_username,
)
from rag.storage.database import SessionLocal
from rag.storage.models import ServiceArtifact, User


_generation_model = None
_verifier_model = None


def _model(temperature: float):
    return init_chat_model(
        model=os.getenv("MODEL"),
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=temperature,
        stream_usage=True,
    )


def _get_generation_model():
    global _generation_model
    if _generation_model is None:
        _generation_model = _model(0.2)
    return _generation_model


def _get_verifier_model():
    global _verifier_model
    if _verifier_model is None:
        _verifier_model = _model(0)
    return _verifier_model


def _tool_payload(tool, arguments: dict) -> dict:
    raw = tool.invoke(arguments)
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def _user_tool_payload(username: str, tool, arguments: dict) -> dict:
    token = set_service_username(username)
    try:
        return _tool_payload(tool, arguments)
    finally:
        reset_service_username(token)


def _source_chunks(payload: dict) -> list[dict]:
    return list(payload.get("context_chunks") or [])


def _source_ids(payload: dict) -> list[str]:
    return list(
        dict.fromkeys(
            chunk.get("chunk_id")
            for chunk in _source_chunks(payload)
            if chunk.get("chunk_id")
        )
    )


def _generate(task_name: str, payload: dict, schema: type[BaseModel]) -> dict:
    prompt = f"""
你是智导（CommercePilot）系统的{task_name} Agent。请严格依据上下文生成结构化结果。

规则：
1. 不得虚构商品参数、库存、价格、优惠、物流时效或售后承诺。
2. 证据不足时写入 limitation，并指出需要人工确认的内容。
3. 表达自然、简洁、合规，不使用绝对化或施压式销售话术。
4. source_chunk_ids 只能来自 context_chunks。

任务上下文：
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()
    result = _get_generation_model().with_structured_output(schema).invoke(
        [{"role": "user", "content": prompt}]
    )
    return result.model_dump() if isinstance(result, BaseModel) else dict(result)


def _verify(task_name: str, content: dict, source_ids: list[str]) -> str:
    prompt = f"""
你是客服质量校验 Agent。检查以下{task_name}是否存在事实编造、无依据承诺、
遗漏客户需求、过度营销或引用不一致。用三条以内中文短句给出结论；
没有问题时输出“校验通过”。

允许的 source_chunk_ids：{source_ids}
内容：{json.dumps(content, ensure_ascii=False)}
""".strip()
    response = _get_verifier_model().invoke(prompt)
    return str(getattr(response, "content", response)).strip()


def _save(
    username: str,
    artifact_type: str,
    title: str,
    prompt: dict,
    content: dict,
    source_ids: list[str],
) -> int | None:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return None
        artifact = ServiceArtifact(
            owner_id=user.id,
            artifact_type=artifact_type,
            title=title,
            prompt=json.dumps(prompt, ensure_ascii=False),
            content_json=content,
            source_chunk_ids=source_ids,
        )
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        return artifact.id
    finally:
        db.close()


def _response(
    *,
    username: str,
    request: Any,
    artifact_type: str,
    title: str,
    task_name: str,
    payload: dict,
    schema: type[BaseModel],
    route: str,
) -> dict:
    content = _generate(task_name, payload, schema)
    source_ids = _source_ids(payload)
    notes = _verify(task_name, content, source_ids)
    saved_id = (
        _save(
            username,
            artifact_type,
            title,
            request.model_dump(),
            content,
            source_ids,
        )
        if request.save
        else None
    )
    return {
        "artifact_type": artifact_type,
        "title": title,
        "content": content,
        "source_chunk_ids": source_ids,
        "source_chunks": _source_chunks(payload),
        "verifier_notes": notes,
        "agent_route": route,
        "saved_artifact_id": saved_id,
    }


def generate_faq(request: Any, username: str) -> dict:
    payload = _user_tool_payload(
        username,
        prepare_faq,
        {key: value for key, value in request.model_dump().items() if key != "save"},
    )
    return _response(
        username=username,
        request=request,
        artifact_type="faq_set",
        title=f"{request.topic} FAQ",
        task_name="FAQ 生成",
        payload=payload,
        schema=FaqSetOutput,
        route="supervisor -> faq_specialist -> verifier",
    )


def generate_sales_script(request: Any, username: str) -> dict:
    payload = _user_tool_payload(
        username,
        prepare_sales_script,
        {key: value for key, value in request.model_dump().items() if key != "save"},
    )
    return _response(
        username=username,
        request=request,
        artifact_type="sales_script",
        title=f"{request.customer_need} 导购话术",
        task_name="导购话术生成",
        payload=payload,
        schema=SalesScriptOutput,
        route="supervisor -> sales_script_specialist -> verifier",
    )


def review_service_reply(request: Any, username: str) -> dict:
    payload = _user_tool_payload(
        username,
        prepare_reply_review,
        {key: value for key, value in request.model_dump().items() if key != "save"},
    )
    return _response(
        username=username,
        request=request,
        artifact_type="reply_review",
        title="客服回复质检",
        task_name="客服回复质检",
        payload=payload,
        schema=ReplyReviewOutput,
        route="supervisor -> quality_reviewer -> verifier",
    )
