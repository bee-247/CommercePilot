"""Central model routing: text, vision, embedding and optional reranker."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from core.config import get_settings
from langchain_openai import ChatOpenAI


def create_chat_model(
    role: Literal["text", "vision"] = "text",
    *,
    enable_thinking: bool | None = None,
    **parameters: Any,
) -> ChatOpenAI:
    settings = get_settings()
    if role not in {"text", "vision"}:
        raise ValueError(f"Unsupported chat model role: {role}")
    model = getattr(settings, f"{role}_llm")
    api_key = getattr(settings, f"{role}_api_key")
    base_url = getattr(settings, f"{role}_base_url")
    if not model or not api_key or not base_url:
        raise ValueError(
            f"请配置 {role.upper()}_LLM、{role.upper()}_API_KEY 和 "
            f"{role.upper()}_BASE_URL"
        )
    forbidden = {"model", "api_key", "base_url", "model_provider"} & parameters.keys()
    if forbidden:
        raise ValueError("模型连接配置必须由四类模型配置统一提供")
    model_name = model.casefold()
    extra_body = dict(parameters.get("extra_body") or {})
    configured_thinking = extra_body.pop("enable_thinking", enable_thinking)

    if model_name.startswith("glm-"):
        temperature = parameters.get("temperature")
        if isinstance(temperature, (int, float)) and temperature <= 0:
            parameters["temperature"] = 0.01

        if configured_thinking is not None and "thinking" not in extra_body:
            extra_body["thinking"] = {
                "type": "enabled" if configured_thinking else "disabled"
            }
        if role == "text" and "thinking" not in extra_body:
            extra_body["thinking"] = {"type": "disabled"}
    elif model_name.startswith("qwen") and configured_thinking is not None:
        # DashScope's OpenAI-compatible Qwen endpoint uses this request field.
        extra_body["enable_thinking"] = configured_thinking

    if extra_body:
        parameters["extra_body"] = extra_body
    else:
        parameters.pop("extra_body", None)
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, **parameters)


def embedding_index_id() -> str:
    """Keep vectors from different models apart, even with the same dimension."""
    settings = get_settings()
    identity = json.dumps(
        [
            settings.embedding_base_url.rstrip("/"),
            settings.embedding_model,
            settings.embedding_dimension,
        ]
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def embedding_collection_name(base_name: str) -> str:
    return f"{base_name}_e{embedding_index_id()}"


def rerank_endpoint() -> str:
    base_url = get_settings().rerank_base_url.strip().rstrip("/")
    if not base_url:
        return ""
    if base_url.endswith("/rerank"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/rerank"
    return f"{base_url}/v1/rerank"
