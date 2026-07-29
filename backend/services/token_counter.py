from __future__ import annotations

import json
import re
from typing import Any


def estimate_tokens(value: Any) -> int:
    """Estimate token count without binding the app to a tokenizer package."""
    text = _to_text(value)
    if not text:
        return 0
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    non_chinese_text = re.sub(r"[\u4e00-\u9fff]", " ", text)
    word_pieces = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", non_chinese_text)
    return len(chinese_chars) + max(1, int(len(word_pieces) * 1.3))


def build_token_usage(
    *,
    request_payload: Any,
    response_payload: Any,
    session_memory: Any | None = None,
    agent_trace: Any | None = None,
    compression_payload: Any | None = None,
    summary: Any | None = None,
    recent_messages: Any | None = None,
    pending_turns: Any | None = None,
) -> dict[str, int]:
    input_tokens = estimate_tokens(request_payload)
    output_tokens = estimate_tokens(response_payload)
    agent_trace_tokens = estimate_tokens(agent_trace)
    compression_payload_tokens = estimate_tokens(compression_payload)
    session_memory_tokens = estimate_tokens(session_memory)
    summary_tokens = estimate_tokens(summary)
    recent_messages_tokens = estimate_tokens(recent_messages)
    pending_turn_tokens = estimate_tokens(pending_turns)
    return {
        "total_tokens": (
            input_tokens
            + output_tokens
            + agent_trace_tokens
            + compression_payload_tokens
            + session_memory_tokens
            + summary_tokens
            + recent_messages_tokens
            + pending_turn_tokens
        ),
    }


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)
