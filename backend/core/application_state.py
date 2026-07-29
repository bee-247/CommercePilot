"""Shared application services for API routers.

The API layer imports these singletons instead of constructing its own copies.
"""

from agents import (
    ChatRouterAgent,
    ConversationContextAgent,
    ConversationReplyAgent,
    ImageUnderstandingAgent,
)
from orchestrator.supervisor import SupervisorOrchestrator
from repositories import ProductRepository
from services.ab_test import ABTestEngine
from services.conversation_memory import ConversationMemoryStore
from services.metrics import MetricsCollector
from services.fusion.sales_rag_service import SalesRagService
from services.vector_indexing import VectorIndexingService


ab_engine = ABTestEngine()
metrics_collector = MetricsCollector()
sales_rag_service = SalesRagService()
supervisor = SupervisorOrchestrator(
    ab_engine=ab_engine,
    sales_rag_service=sales_rag_service,
)
vector_indexing_service = VectorIndexingService()
image_understanding_agent = ImageUnderstandingAgent()
conversation_context_agent = ConversationContextAgent()
chat_router_agent = ChatRouterAgent()
conversation_reply_agent = ConversationReplyAgent()
product_repository = ProductRepository()
conversation_memory = ConversationMemoryStore()
