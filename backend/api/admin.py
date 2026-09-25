"""Administrative and indexing API routes."""

from core.application_state import (
    ab_engine,
    agent_evaluation_store,
    metrics_collector,
    vector_indexing_service,
)
from core.auth import require_admin
from fastapi import APIRouter, Depends

router = APIRouter(
    prefix="/api/v1",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


@router.get("/experiments")
async def get_experiments():
    """查看所有A/B实验状态"""
    experiments = {}
    for exp_id, exp in ab_engine.experiments.items():
        experiments[exp_id] = {
            "name": exp.name,
            "enabled": exp.enabled,
            "groups": [
                {
                    "name": g.name,
                    "weight": g.weight,
                    "config": g.config,
                    "successes": g.successes,
                    "failures": g.failures,
                }
                for g in exp.groups
            ],
            "stats": ab_engine.get_stats(exp_id),
        }
    return experiments


@router.get("/metrics")
async def get_metrics():
    """查看系统监控指标"""
    return {
        "agents": metrics_collector.get_agent_stats(),
        "business": metrics_collector.get_business_stats(),
    }


@router.get("/agent-evaluations")
def get_agent_evaluations(days: int = 30):
    """查看持久化的 Agent 运行、业务质量与 Replan 聚合指标。"""
    return agent_evaluation_store.get_summary(days=days)


@router.post("/vector-index/products")
async def index_products(limit: int = 1000):
    """Encode product title/name + description into the product vector collection."""
    return await vector_indexing_service.index_products(limit=limit)


@router.post("/vector-index/users/{user_id}")
async def index_user(user_id: str, history_limit: int = 50):
    """Encode a user's product interaction history into the user vector collection."""
    success = await vector_indexing_service.index_user(
        user_id=user_id,
        history_limit=history_limit,
    )
    return {"user_id": user_id, "indexed": success}


@router.post("/experiments/{experiment_id}/outcome")
async def record_outcome(experiment_id: str, group: str, success: bool):
    """记录A/B测试结果,更新Thompson Sampling"""
    ab_engine.record_outcome(experiment_id, group, success)
    return {"status": "recorded"}
