"""Central API router registration."""

from fastapi import APIRouter

from api import admin, health, rag, sales_chat


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(sales_chat.router)
api_router.include_router(admin.router)
api_router.include_router(rag.router)
