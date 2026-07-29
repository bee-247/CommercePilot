"""CommercePilot FastAPI entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from core.env import load_project_env

load_project_env()

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import api_router
from core.auth import ensure_bootstrap_admin
from core.config import get_settings
from database import init_db
from database.seed import seed_demo_catalog


logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_bootstrap_admin()
    if settings.seed_demo_data:
        seeded = seed_demo_catalog()
        if seeded:
            logger.info("demo_catalog.seeded", product_count=seeded)
    logger.info("app.startup", model=settings.llm_model)
    yield
    logger.info("app.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="智导 · CommercePilot",
        description="Intelligent shopping guide and customer-service RAG system",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
