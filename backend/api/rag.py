"""Combined RAG router assembled from focused route groups."""

from fastapi import APIRouter

from api import account, customer_service, documents, rag_chat


router = APIRouter(prefix="/api/rag")
router.include_router(account.router)
router.include_router(rag_chat.router)
router.include_router(documents.router)
router.include_router(customer_service.router)
