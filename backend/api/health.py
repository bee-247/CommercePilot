"""Basic backend status routes."""

import asyncio
import os
from collections.abc import Callable
from typing import Any

from core.config import get_settings
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

router = APIRouter(tags=["health"])


@router.get("/")
async def root_status():
    settings = get_settings()
    return {
        "name": "CommercePilot backend",
        "status": "ok",
        "model": settings.text_llm,
    }


@router.get("/health")
async def health():
    settings = get_settings()
    return {"status": "healthy", "model": settings.text_llm}


def _check_sql(engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


def _check_configuration() -> None:
    settings = get_settings()
    missing = []
    if len(os.getenv("JWT_SECRET_KEY", "")) < 32:
        missing.append("JWT_SECRET_KEY")
    model_fields = {
        "text": ("llm", "api_key", "base_url"),
        "vision": ("llm", "api_key", "base_url"),
        "embedding": ("model", "api_key", "base_url"),
    }
    for role, fields in model_fields.items():
        for field in fields:
            if not getattr(settings, f"{role}_{field}"):
                missing.append(f"{role.upper()}_{field.upper()}")
    if missing:
        raise ValueError("missing or invalid: " + ", ".join(missing))


async def _run_check(name: str, check: Callable[[], Any]) -> tuple[str, dict[str, str]]:
    try:
        await asyncio.wait_for(asyncio.to_thread(check), timeout=3.0)
        return name, {"status": "ready"}
    except TimeoutError:
        return name, {"status": "unavailable", "reason": "timeout"}
    except Exception as exc:
        return name, {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {str(exc)[:160]}",
        }


@router.get("/health/ready")
async def readiness():
    """Check every external component required by the complete workflow."""
    from database import product_engine, user_engine
    from rag.storage.cache import cache
    from rag.storage.database import engine as rag_engine
    from rag.storage.milvus_client import MilvusManager

    milvus = MilvusManager()
    checks = await asyncio.gather(
        _run_check("configuration", _check_configuration),
        _run_check("product_database", lambda: _check_sql(product_engine)),
        _run_check("user_database", lambda: _check_sql(user_engine)),
        _run_check("rag_database", lambda: _check_sql(rag_engine)),
        _run_check("redis", lambda: cache._get_client().ping()),
        _run_check(
            "milvus",
            lambda: milvus._get_client().list_collections(),
        ),
    )
    components = dict(checks)
    ready = all(value["status"] == "ready" for value in components.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "components": components,
            "models": {
                "text": get_settings().text_llm,
                "vision": get_settings().vision_llm,
                "embedding": get_settings().embedding_model,
                "reranker": get_settings().rerank_model,
            },
        },
    )
