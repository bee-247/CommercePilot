from __future__ import annotations

import json
from typing import Any


def strip_json_fence(raw: str) -> str:
    cleaned = str(raw or "").strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("\n", 1)
        if len(parts) == 1:
            return ""
        cleaned = parts[1].rsplit("```", 1)[0].strip()
    return cleaned


def parse_json_object(raw: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    cleaned = strip_json_fence(raw)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return default or {}
    return data if isinstance(data, dict) else (default or {})


def parse_json_array(raw: str, default: list[Any] | None = None) -> list[Any]:
    data = try_parse_json_array(raw)
    if data is None:
        return default or []
    return data


def try_parse_json_array(raw: str) -> list[Any] | None:
    cleaned = strip_json_fence(raw)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None
