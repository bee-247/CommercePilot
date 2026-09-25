"""Shared application services for API routers.

The API layer imports these singletons instead of constructing its own copies.
"""

from agents import (
    ImageUnderstandingAgent,
)
from agents.runtime import get_quality_reviewer, get_response_agent, get_understanding_agent
from core.agent_config import get_agent_system_config
from orchestrator.adaptive_mesh import AdaptiveAgentMeshOrchestrator
from orchestrator.supervisor import SupervisorOrchestrator
from repositories import ProductRepository
from services.ab_test import ABTestEngine
from services.agent_evaluation import AgentEvaluationStore
from services.conversation_memory import ConversationMemoryStore
from services.fusion.sales_rag_service import SalesRagService
from services.metrics import MetricsCollector
from services.vector_indexing import VectorIndexingService

ab_engine = ABTestEngine()
metrics_collector = MetricsCollector()
sales_rag_service = SalesRagService()
agent_evaluation_store = AgentEvaluationStore()
agent_system_config = get_agent_system_config()
supervisor = SupervisorOrchestrator(
    ab_engine=ab_engine,
    sales_rag_service=sales_rag_service,
)
adaptive_agent_mesh = AdaptiveAgentMeshOrchestrator(
    workflow=supervisor,
    ab_engine=ab_engine,
    sales_rag_service=sales_rag_service,
    agent_config=agent_system_config,
    evaluation_store=agent_evaluation_store,
)
recommendation_orchestrator = (
    adaptive_agent_mesh
    if agent_system_config.orchestration.mode.strip().lower() == "mesh"
    else supervisor
)
vector_indexing_service = VectorIndexingService()
image_understanding_agent = ImageUnderstandingAgent()
conversation_understanding_agent = get_understanding_agent()
response_generation_agent = get_response_agent()
quality_reviewer_agent = get_quality_reviewer()
product_repository = ProductRepository()
conversation_memory = ConversationMemoryStore()
