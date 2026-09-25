"""Structured generation and verification for customer-service tools."""

from __future__ import annotations

import json
from typing import Any

from agents.runtime import get_quality_reviewer, get_response_agent
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
from pydantic import BaseModel
from rag.storage.database import SessionLocal
from rag.storage.models import ServiceArtifact, User

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


def _generate(task_name: str, payload: dict, schema: type[BaseModel], *, feedback=None, previous_content=None) -> dict:
    if schema is ReplyReviewOutput:
        return get_quality_reviewer().generate_report(payload, feedback)
    return get_response_agent().generate_structured(
        task_name, payload, schema, feedback=feedback, previous_content=previous_content,
    )


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
    reviewer = get_quality_reviewer()
    review_args = {
        "mode": "customer_service",
        "context": {"task": task_name, "requirements": payload.get("requirements"), "inputs": payload.get("inputs")},
        "evidence": _source_chunks(payload),
    }
    audit = reviewer.review_sync(draft=content, **review_args)
    if audit.success and not audit.passed and audit.retry_recommended:
        content = _generate(
            task_name, payload, schema,
            feedback=[issue.model_dump() for issue in audit.issues], previous_content=content,
        )
        audit = reviewer.review_sync(draft=content, **review_args)
    if not audit.success or not audit.passed:
        raise ValueError("生成内容未通过质量审核，未保存或交付草稿；请补充资料后重试")
    notes = "校验通过"
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
        route="response-generation(faq) -> quality-reviewer",
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
        route="response-generation(sales_script) -> quality-reviewer",
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
        route="quality-reviewer(reply_review) -> quality-reviewer(final_check)",
    )
