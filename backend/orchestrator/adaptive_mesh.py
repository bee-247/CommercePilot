"""Adaptive Agent Mesh demo built on top of the existing sales agents.

The mesh keeps domain behavior in the existing agents, while replacing the
hard-coded execution order with a small dependency-aware task graph.  It is a
single-process demo: the blackboard and broker are intentionally in memory.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog
from agents.product_rec_agent import ProductRecAgent
from core.agent_config import (
    AgentSystemConfig,
    ModelProfileConfig,
    get_agent_system_config,
)
from core.config import get_settings
from langchain_core.messages import HumanMessage, SystemMessage
from core.model_clients import create_chat_model
from models.schemas import (
    AgentResult,
    Product,
    RecommendationRequest,
    RecommendationResponse,
)
from pydantic import BaseModel, Field
from services.ab_test import ABTestEngine
from services.agent_evaluation import (
    QUALITY_CREDIT_CAPABILITIES,
    AgentEvaluationStore,
)
from services.execution_progress import report_progress
from services.fusion.sales_rag_service import SalesRagService

from .supervisor import SupervisorOrchestrator

logger = structlog.get_logger()


class MeshTask(BaseModel):
    """One capability request in the dynamically executed task DAG."""

    task_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    capability: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    depends_on: list[str] = Field(default_factory=list)
    input: dict[str, Any] = Field(default_factory=dict)


class MeshExecutionPlan(BaseModel):
    """A validated group of tasks that may be appended at runtime."""

    phase: str
    reason: str
    tasks: list[MeshTask]
    planner: str = "rule"
    fallback_reason: str | None = None


class ReplanDecision(BaseModel):
    """Structured LLM output accepted by the replan safety gate."""

    action: Literal["append_tasks", "finish"]
    reason: str = Field(min_length=1, max_length=500)
    tasks: list[MeshTask] = Field(default_factory=list)


class InitialPlanDecision(BaseModel):
    """Structured LLM output for the initial recommendation DAG."""

    reason: str = Field(min_length=1, max_length=500)
    tasks: list[MeshTask] = Field(min_length=1)


class BrokerAgentAssessment(BaseModel):
    """LLM assessment of one registered Agent for the current task."""

    agent_id: str = Field(min_length=1, max_length=100)
    can_handle: bool = True
    scenario_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)


class BrokerRoutingDecision(BaseModel):
    """Structured semantic assessments consumed by the metric Broker."""

    assessments: list[BrokerAgentAssessment] = Field(min_length=1)


class AgentCard(BaseModel):
    """Capability plus measurable defaults used before runtime data exists."""

    agent_id: str
    capabilities: list[str]
    historical_success: float = Field(default=0.95, ge=0.0, le=1.0)
    average_latency_ms: float = Field(default=500.0, ge=0.0)
    cost_level: float = Field(default=1.0, ge=0.0, le=5.0)
    max_concurrency: int = Field(default=4, ge=1)


class AgentBid(BaseModel):
    """An agent's dynamic assessment of its fit for the current task."""

    can_handle: bool = True
    scenario_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)


TaskExecutor = Callable[[MeshTask, "MeshBlackboard"], Awaitable[dict[str, Any]]]
ScenarioBidder = Callable[[MeshTask, "MeshBlackboard"], AgentBid]


@dataclass
class _RegisteredAgent:
    card: AgentCard
    executor: TaskExecutor
    bidder: ScenarioBidder
    calls: int = 0
    successes: int = 0
    failures: int = 0
    business_feedback_count: int = 0
    business_successes: int = 0
    active_tasks: int = 0
    observed_latency_ms: float | None = None


class AgentRegistry:
    """Registry using measurable metrics and agent-provided scenario bids."""

    default_weights = {
        "historical_success": 0.30,
        "latency": 0.20,
        "cost": 0.10,
        "scenario": 0.40,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        success_prior_calls: float = 5.0,
        latency_ceiling_ms: float = 5000.0,
    ) -> None:
        self._agents: dict[str, _RegisteredAgent] = {}
        self.weights = {**self.default_weights, **(weights or {})}
        self.success_prior_calls = max(success_prior_calls, 0.0)
        self.latency_ceiling_ms = max(latency_ceiling_ms, 1.0)

    def register(
        self,
        card: AgentCard,
        executor: TaskExecutor,
        *,
        bidder: ScenarioBidder | None = None,
    ) -> None:
        self._agents[card.agent_id] = _RegisteredAgent(
            card=card,
            executor=executor,
            bidder=bidder or self._capability_bid,
        )

    def has_capability(self, capability: str) -> bool:
        return any(
            capability in agent.card.capabilities
            for agent in self._agents.values()
        )

    def select(
        self,
        capability: str,
        *,
        task: MeshTask | None = None,
        blackboard: MeshBlackboard | None = None,
    ) -> tuple[_RegisteredAgent, float]:
        selected, score, _ranking = self.select_with_ranking(
            capability,
            task=task,
            blackboard=blackboard,
        )
        return selected, score

    def select_with_ranking(
        self,
        capability: str,
        *,
        task: MeshTask | None = None,
        blackboard: MeshBlackboard | None = None,
        semantic_assessments: dict[str, BrokerAgentAssessment] | None = None,
        routing_mode: str = "metric",
        routing_fallback_reason: str | None = None,
    ) -> tuple[_RegisteredAgent, float, list[dict[str, Any]]]:
        candidates = [
            agent
            for agent in self._agents.values()
            if capability in agent.card.capabilities
        ]
        if not candidates:
            raise LookupError(f"没有 Agent 注册能力: {capability}")

        resolved_task = task or MeshTask(
            task_id="broker_selection",
            capability=capability,
        )
        resolved_blackboard = blackboard or MeshBlackboard(
            request=RecommendationRequest(user_id="broker-selection"),
            request_id="broker-selection",
        )
        scored_candidates = []
        for agent in candidates:
            rule_bid = self._request_bid(
                agent,
                resolved_task,
                resolved_blackboard,
            )
            llm_assessment = (semantic_assessments or {}).get(agent.card.agent_id)
            effective_bid = self._merge_bids(rule_bid, llm_assessment)
            scored_candidates.append(
                (
                    self._broker_score(agent, effective_bid),
                    effective_bid,
                    rule_bid,
                    llm_assessment,
                    agent,
                )
            )
        ranked = sorted(
            scored_candidates,
            key=lambda item: (item[1].can_handle, item[0]["total"]),
            reverse=True,
        )
        selected_breakdown, selected_bid, _rule_bid, _assessment, selected = ranked[0]
        if not selected_bid.can_handle:
            raise LookupError(f"所有 Agent 都拒绝能力任务: {capability}")
        ranking = [
            {
                "agent_id": agent.card.agent_id,
                "score": breakdown["total"] if bid.can_handle else 0.0,
                "selected": agent.card.agent_id == selected.card.agent_id,
                "components": breakdown["components"],
                "bid": bid.model_dump(),
                "rule_bid": rule_bid.model_dump(),
                "llm_assessment": (
                    llm_assessment.model_dump() if llm_assessment else None
                ),
                "routing_mode": routing_mode,
                "routing_fallback_reason": routing_fallback_reason,
                "active_tasks": agent.active_tasks,
            }
            for breakdown, bid, rule_bid, llm_assessment, agent in ranked
        ]
        return selected, selected_breakdown["total"], ranking

    def routing_candidates(
        self,
        capability: str,
        *,
        task: MeshTask,
        blackboard: MeshBlackboard,
    ) -> list[dict[str, Any]]:
        """Return safe, serializable candidate facts for LLM assessment."""
        candidates = [
            agent
            for agent in self._agents.values()
            if capability in agent.card.capabilities
        ]
        return [
            {
                "agent_id": agent.card.agent_id,
                "capabilities": agent.card.capabilities,
                "rule_bid": self._request_bid(agent, task, blackboard).model_dump(),
            }
            for agent in sorted(candidates, key=lambda item: item.card.agent_id)
        ]

    def mark_started(self, agent_id: str) -> None:
        self._agents[agent_id].active_tasks += 1

    def mark_finished(
        self,
        agent_id: str,
        *,
        success: bool,
        latency_ms: float,
    ) -> None:
        agent = self._agents[agent_id]
        agent.active_tasks = max(0, agent.active_tasks - 1)
        agent.calls += 1
        if success:
            agent.successes += 1
        else:
            agent.failures += 1
        if agent.observed_latency_ms is None:
            agent.observed_latency_ms = latency_ms
        else:
            agent.observed_latency_ms = (
                agent.observed_latency_ms * 0.8 + latency_ms * 0.2
            )

    def mark_business_outcome(self, agent_id: str, *, success: bool) -> None:
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        agent.business_feedback_count += 1
        if success:
            agent.business_successes += 1

    def restore_runtime(self, stats: dict[str, dict[str, Any]]) -> None:
        """Hydrate process-local routing metrics from persistent events."""
        for agent_id, values in stats.items():
            agent = self._agents.get(agent_id)
            if agent is None:
                continue
            agent.calls = max(0, int(values.get("calls") or 0))
            agent.successes = max(0, int(values.get("successes") or 0))
            agent.failures = max(0, int(values.get("failures") or 0))
            agent.business_feedback_count = max(
                0,
                int(values.get("business_feedback_count") or 0),
            )
            agent.business_successes = max(
                0,
                int(values.get("business_successes") or 0),
            )
            observed_latency = values.get("observed_latency_ms")
            agent.observed_latency_ms = (
                max(0.0, float(observed_latency))
                if observed_latency is not None
                else None
            )

    def cards(self) -> list[dict[str, Any]]:
        return [
            {
                **agent.card.model_dump(),
                "runtime": {
                    "calls": agent.calls,
                    "successes": agent.successes,
                    "failures": agent.failures,
                    "business_feedback_count": agent.business_feedback_count,
                    "business_successes": agent.business_successes,
                    "business_success_rate": round(
                        agent.business_successes / agent.business_feedback_count,
                        4,
                    )
                    if agent.business_feedback_count
                    else None,
                    "active_tasks": agent.active_tasks,
                    "observed_latency_ms": (
                        round(agent.observed_latency_ms, 1)
                        if agent.observed_latency_ms is not None
                        else None
                    ),
                },
            }
            for agent in sorted(
                self._agents.values(),
                key=lambda item: item.card.agent_id,
            )
        ]

    def _broker_score(
        self,
        agent: _RegisteredAgent,
        bid: AgentBid,
    ) -> dict[str, Any]:
        card = agent.card
        observed_calls = (
            agent.business_feedback_count
            if agent.business_feedback_count
            else agent.calls
        )
        observed_successes = (
            agent.business_successes
            if agent.business_feedback_count
            else agent.successes
        )
        prior_successes = card.historical_success * self.success_prior_calls
        denominator = observed_calls + self.success_prior_calls
        success_score = (
            (observed_successes + prior_successes) / denominator
            if denominator
            else card.historical_success
        )
        latency = (
            agent.observed_latency_ms
            if agent.observed_latency_ms is not None
            else card.average_latency_ms
        )
        latency_score = 1.0 - min(latency / self.latency_ceiling_ms, 1.0)
        cost_score = 1.0 - min(card.cost_level / 5.0, 1.0)
        components = {
            "historical_success": round(success_score, 4),
            "latency": round(latency_score, 4),
            "cost": round(cost_score, 4),
            "scenario": round(bid.scenario_score, 4),
        }
        total = (
            success_score * self.weights["historical_success"]
            + latency_score * self.weights["latency"]
            + cost_score * self.weights["cost"]
            + bid.scenario_score * self.weights["scenario"]
        )
        return {"total": round(total, 4), "components": components}

    @staticmethod
    def _request_bid(
        agent: _RegisteredAgent,
        task: MeshTask,
        blackboard: MeshBlackboard,
    ) -> AgentBid:
        try:
            return AgentBid.model_validate(agent.bidder(task, blackboard))
        except Exception as exc:
            return AgentBid(
                can_handle=False,
                scenario_score=0.0,
                reason=f"Agent 场景报价失败: {exc}",
            )

    @staticmethod
    def _merge_bids(
        rule_bid: AgentBid,
        llm_assessment: BrokerAgentAssessment | None,
    ) -> AgentBid:
        if llm_assessment is None:
            return rule_bid
        return AgentBid(
            can_handle=rule_bid.can_handle and llm_assessment.can_handle,
            scenario_score=round(
                rule_bid.scenario_score * 0.4
                + llm_assessment.scenario_score * 0.6,
                4,
            ),
            reason=(
                f"规则报价：{rule_bid.reason}；"
                f"LLM 场景评估：{llm_assessment.reason}"
            ),
        )

    @staticmethod
    def _capability_bid(
        task: MeshTask,
        _blackboard: MeshBlackboard,
    ) -> AgentBid:
        return AgentBid(
            can_handle=True,
            scenario_score=1.0,
            reason=f"Agent 确认可执行 capability={task.capability}",
        )

    def runtime_policy(self) -> dict[str, float]:
        return {
            "success_prior_calls": self.success_prior_calls,
            "latency_ceiling_ms": self.latency_ceiling_ms,
        }


@dataclass
class MeshBlackboard:
    """Request-scoped shared state and execution trace."""

    request: RecommendationRequest
    request_id: str
    values: dict[str, Any] = field(default_factory=dict)
    completed_tasks: set[str] = field(default_factory=set)
    task_traces: list[dict[str, Any]] = field(default_factory=list)
    waves: list[list[str]] = field(default_factory=list)
    plans: list[dict[str, Any]] = field(default_factory=list)
    business_feedback_applied: bool = False
    replan: dict[str, Any] = field(
        default_factory=lambda: {
            "triggered": False,
            "reason": "judge_passed_or_retry_not_requested",
            "added_tasks": [],
            "planner": "rule",
            "fallback_reason": None,
        }
    )

    def publish(self, values: dict[str, Any]) -> None:
        self.values.update(values)

    def add_plan(self, plan: MeshExecutionPlan) -> None:
        self.plans.append(plan.model_dump())

    def trace(self, registry: AgentRegistry) -> dict[str, Any]:
        return {
            "architecture": "adaptive_agent_mesh_demo",
            "planner": "structured_llm_planner_and_replanner_with_rule_fallback",
            "broker": "llm_semantic_assessment_with_metric_routing",
            "blackboard": "request_scoped_in_memory",
            "plans": self.plans,
            "waves": self.waves,
            "tasks": self.task_traces,
            "replan": self.replan,
            "blackboard_keys": sorted(self.values),
            "completed_task_ids": sorted(self.completed_tasks),
            "agent_registry": registry.cards(),
            "broker_weights": dict(registry.weights),
            "broker_runtime_policy": registry.runtime_policy(),
        }


class AdaptivePlanBuilder:
    """Provide deterministic fallback plans for both planning stages."""

    def __init__(self, max_replans: int = 1) -> None:
        self.max_replans = max(0, max_replans)

    def initial_plan(
        self,
        request: RecommendationRequest,
        *,
        include_knowledge: bool,
    ) -> MeshExecutionPlan:
        tasks = [MeshTask(task_id="product_recall", capability="product_recall")]
        dependencies = ["product_recall"]
        if include_knowledge:
            tasks.append(MeshTask(
                task_id="knowledge_retrieval", capability="knowledge_retrieval",
                depends_on=["product_recall"],
            ))
            dependencies.append("knowledge_retrieval")
        tasks.extend([
            MeshTask(task_id="response_synthesis", capability="response_synthesis", depends_on=dependencies),
            MeshTask(task_id="quality_review", capability="quality_review", depends_on=["response_synthesis"]),
        ])
        return MeshExecutionPlan(
            phase="initial", reason="召回并规则过滤、按需取证、评估并生成回复，最后统一审核", tasks=tasks,
        )

    def continuation_plan(self, audit_result: AgentResult) -> MeshExecutionPlan:
        should_replan = (
            self.max_replans > 0 and audit_result.success
            and not getattr(audit_result, "passed", False)
            and getattr(audit_result, "retry_recommended", False)
        )
        if should_replan:
            return MeshExecutionPlan(
                phase="judge_replan", reason="按最终回复审核意见修订，并复审新回复",
                tasks=[
                    MeshTask(task_id="response_revision", capability="response_revision", depends_on=["quality_review"]),
                    MeshTask(task_id="quality_recheck", capability="quality_recheck", depends_on=["response_revision"]),
                ], planner="rule_fallback",
            )
        return MeshExecutionPlan(phase="completion", reason="无需继续修订；交付前应用审核结果", tasks=[])


INITIAL_PLANNER_SYSTEM_PROMPT = """为电商推荐生成最短可执行任务DAG，输出结构化结果。
1. 每种能力最多一次，必须包含product_recall、response_synthesis、quality_review。
2. product_recall是根节点，内部已完成库存与硬约束过滤。
3. knowledge_retrieval仅在知识库可用且需要证据时加入，依赖product_recall。
4. response_synthesis内部进行候选评估并生成最终回复，依赖product_recall和存在的knowledge_retrieval。
5. quality_review必须依赖response_synthesis且是唯一终止节点；禁止审核后再生成未经审核的回复。
6. 只能使用allowed_capabilities，禁止循环依赖，节点数不得超过max_tasks。
"""


class InitialPlanSafetyValidator:
    """Validate an LLM-authored initial plan before execution."""

    allowed_capabilities = {"product_recall", "knowledge_retrieval", "response_synthesis", "quality_review"}
    required_capabilities = {"product_recall", "response_synthesis", "quality_review"}

    def __init__(self, registry: AgentRegistry, max_tasks: int = 6) -> None:
        self.registry = registry
        self.max_tasks = max(1, max_tasks)

    def to_plan(
        self,
        decision: InitialPlanDecision,
        *,
        knowledge_available: bool,
    ) -> MeshExecutionPlan:
        tasks = decision.tasks
        if len(tasks) > self.max_tasks:
            raise ValueError(f"初始计划最多允许 {self.max_tasks} 个任务")

        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("初始计划包含重复 task_id")

        all_task_ids = set(task_ids)
        by_id = {task.task_id: task for task in tasks}
        by_capability: dict[str, list[MeshTask]] = {}
        for task in tasks:
            if task.capability not in self.allowed_capabilities:
                raise ValueError(f"初始计划使用了不安全能力: {task.capability}")
            if not self.registry.has_capability(task.capability):
                raise ValueError(f"初始计划能力未注册: {task.capability}")
            missing = set(task.depends_on) - all_task_ids
            if missing:
                raise ValueError(
                    f"初始任务 {task.task_id} 依赖不存在: {sorted(missing)}"
                )
            if task.task_id in task.depends_on:
                raise ValueError(f"初始任务 {task.task_id} 不能依赖自身")
            by_capability.setdefault(task.capability, []).append(task)

        duplicate_capabilities = sorted(
            capability
            for capability, capability_tasks in by_capability.items()
            if len(capability_tasks) > 1
        )
        if duplicate_capabilities:
            raise ValueError(f"初始计划包含重复能力: {duplicate_capabilities}")
        missing_capabilities = sorted(
            self.required_capabilities - set(by_capability)
        )
        if missing_capabilities:
            raise ValueError(f"初始计划缺少必需能力: {missing_capabilities}")
        if "knowledge_retrieval" in by_capability and not knowledge_available:
            raise ValueError("当前没有可用知识库，不能规划 knowledge_retrieval")

        self._reject_cycles(by_id)
        recall = by_capability["product_recall"][0]
        synthesis = by_capability["response_synthesis"][0]
        judge = by_capability["quality_review"][0]
        if recall.depends_on:
            raise ValueError("product_recall 必须是根节点")
        self._require_dependency(synthesis, recall.task_id, by_id)
        knowledge_tasks = by_capability.get("knowledge_retrieval", [])
        if knowledge_tasks:
            knowledge = knowledge_tasks[0]
            self._require_dependency(knowledge, recall.task_id, by_id)
            self._require_dependency(synthesis, knowledge.task_id, by_id)
        self._require_dependency(judge, synthesis.task_id, by_id)

        depended_on = {
            dependency for task in tasks for dependency in task.depends_on
        }
        if all_task_ids - depended_on != {judge.task_id}:
            raise ValueError("quality_review 必须是唯一终止节点")

        canonical_ids = {task.task_id: task.capability for task in tasks}
        tasks = [task.model_copy(update={
            "task_id": task.capability,
            "depends_on": [canonical_ids[item] for item in task.depends_on],
        }) for task in tasks]
        return MeshExecutionPlan(
            phase="initial",
            reason=decision.reason,
            tasks=tasks,
            planner="llm_planner",
        )

    @staticmethod
    def _reject_cycles(by_id: dict[str, MeshTask]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise ValueError("初始计划包含循环依赖")
            visiting.add(task_id)
            for dependency in by_id[task_id].depends_on:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in by_id:
            visit(task_id)

    @staticmethod
    def _require_dependency(
        task: MeshTask,
        required_task_id: str,
        by_id: dict[str, MeshTask],
    ) -> None:
        def reaches(task_id: str, seen: set[str]) -> bool:
            if task_id == required_task_id:
                return True
            if task_id in seen:
                return False
            return any(
                reaches(dependency, seen | {task_id})
                for dependency in by_id[task_id].depends_on
            )

        if not any(reaches(dependency, set()) for dependency in task.depends_on):
            raise ValueError(
                f"初始任务 {task.task_id} 必须依赖 {required_task_id}"
            )


class LLMPlanner:
    """Generate the initial DAG with an LLM and rules as fallback."""

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        fallback_planner: AdaptivePlanBuilder,
        enabled: bool = True,
        max_tasks: int = 6,
        structured_model: Any | None = None,
        model_config: ModelProfileConfig | None = None,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.fallback_planner = fallback_planner
        self.validator = InitialPlanSafetyValidator(registry, max_tasks=max_tasks)
        self.model = structured_model or self._build_model(enabled, model_config)
        self.timeout_seconds = max(0.1, timeout_seconds)

    async def initial_plan(
        self,
        request: RecommendationRequest,
        *,
        include_knowledge: bool,
    ) -> MeshExecutionPlan:
        if self.model is None:
            return self._fallback(
                request,
                include_knowledge,
                "LLM Planner 未启用或未配置模型凭证",
            )

        payload = {
            "request": {
                "scene": request.scene,
                "query": request.context.get("query")
                or request.context.get("keyword")
                or "",
                "budget": request.context.get("budget"),
                "constraints": request.context.get("constraints") or [],
                "shopping_goals": request.context.get("shopping_goals") or [],
                "categories": request.context.get("categories") or [],
                "avoid_categories": (
                    request.context.get("avoid_categories") or []
                ),
                "image_summary": request.context.get("image_summary") or "",
            },
            "allowed_capabilities": sorted(
                capability
                for capability in self.validator.allowed_capabilities
                if self.validator.registry.has_capability(capability)
            ),
            "knowledge_available": include_knowledge,
            "limits": {"max_tasks": self.validator.max_tasks},
        }
        try:
            raw_decision = await asyncio.wait_for(
                self.model.ainvoke(
                    [
                        SystemMessage(content=INITIAL_PLANNER_SYSTEM_PROMPT),
                        HumanMessage(
                            content=json.dumps(
                                payload,
                                ensure_ascii=False,
                                default=str,
                            )
                        ),
                    ]
                ),
                timeout=self.timeout_seconds,
            )
            decision = (
                raw_decision
                if isinstance(raw_decision, InitialPlanDecision)
                else InitialPlanDecision.model_validate(raw_decision)
            )
            return self.validator.to_plan(
                decision,
                knowledge_available=include_knowledge,
            )
        except Exception as exc:
            logger.warning("agent_mesh.planner_fallback", error=str(exc))
            return self._fallback(request, include_knowledge, str(exc))

    @staticmethod
    def _build_model(
        enabled: bool,
        model_config: ModelProfileConfig | None,
    ) -> Any | None:
        settings = get_settings()
        if not enabled or not settings.text_api_key:
            return None
        resolved = model_config or ModelProfileConfig(
            temperature=0.0,
            max_tokens=min(settings.llm_max_tokens, 1200),
        )
        model = create_chat_model(
            temperature=resolved.temperature,
            max_tokens=resolved.max_tokens,
            enable_thinking=False,
        )
        return model.with_structured_output(InitialPlanDecision)

    def _fallback(
        self,
        request: RecommendationRequest,
        include_knowledge: bool,
        reason: str,
    ) -> MeshExecutionPlan:
        plan = self.fallback_planner.initial_plan(
            request,
            include_knowledge=include_knowledge,
        )
        return plan.model_copy(
            update={"planner": "rule_fallback", "fallback_reason": reason}
        )


BROKER_SYSTEM_PROMPT = """你是电商 Agent Mesh 的 Broker 语义评估器。
请根据当前任务、用户请求和候选 Agent 的能力，对每个候选 Agent 逐一评估。

要求：
1. 只能评估输入提供的 candidate_agents，不能添加、删除或改名。
2. scenario_score 表示 Agent 与当前任务场景的匹配度，范围为 0 到 1。
3. can_handle=false 只用于能力或输入明确不匹配的情况。
4. 不负责最终选择；成功率、延迟、成本和并发由系统统一计算。
5. reason 使用一句简短结论，不输出思维过程。
"""


class HybridBroker:
    """Blend LLM semantic assessments with deterministic routing metrics."""

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        enabled: bool = True,
        structured_model: Any | None = None,
        model_config: ModelProfileConfig | None = None,
        timeout_seconds: float = 6.0,
    ) -> None:
        self.registry = registry
        self.model = structured_model or self._build_model(enabled, model_config)
        self.timeout_seconds = max(0.1, timeout_seconds)

    async def select_with_ranking(
        self,
        capability: str,
        *,
        task: MeshTask,
        blackboard: MeshBlackboard,
    ) -> tuple[_RegisteredAgent, float, list[dict[str, Any]]]:
        candidates = self.registry.routing_candidates(
            capability,
            task=task,
            blackboard=blackboard,
        )
        if len(candidates) <= 1:
            return self.registry.select_with_ranking(
                capability,
                task=task,
                blackboard=blackboard,
                routing_mode="single_candidate_metric",
            )
        if self.model is None:
            return self._fallback(
                capability,
                task,
                blackboard,
                "LLM Broker 未启用或未配置模型凭证",
            )

        payload = {
            "task": task.model_dump(),
            "request": {
                "scene": blackboard.request.scene,
                "query": blackboard.values.get("guide_query", ""),
                "budget": blackboard.request.context.get("budget"),
                "constraints": (
                    blackboard.request.context.get("constraints") or []
                ),
                "shopping_goals": (
                    blackboard.request.context.get("shopping_goals") or []
                ),
            },
            "available_blackboard_keys": sorted(blackboard.values),
            "candidate_agents": candidates,
        }
        try:
            raw_decision = await asyncio.wait_for(
                self.model.ainvoke(
                    [
                        SystemMessage(content=BROKER_SYSTEM_PROMPT),
                        HumanMessage(
                            content=json.dumps(
                                payload,
                                ensure_ascii=False,
                                default=str,
                            )
                        ),
                    ]
                ),
                timeout=self.timeout_seconds,
            )
            decision = (
                raw_decision
                if isinstance(raw_decision, BrokerRoutingDecision)
                else BrokerRoutingDecision.model_validate(raw_decision)
            )
            assessments = self._validate_assessments(decision, candidates)
            return self.registry.select_with_ranking(
                capability,
                task=task,
                blackboard=blackboard,
                semantic_assessments=assessments,
                routing_mode="llm_semantic_plus_metrics",
            )
        except Exception as exc:
            logger.warning("agent_mesh.broker_fallback", error=str(exc))
            return self._fallback(capability, task, blackboard, str(exc))

    @staticmethod
    def _validate_assessments(
        decision: BrokerRoutingDecision,
        candidates: list[dict[str, Any]],
    ) -> dict[str, BrokerAgentAssessment]:
        expected_ids = {str(candidate["agent_id"]) for candidate in candidates}
        assessment_ids = [item.agent_id for item in decision.assessments]
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError("LLM Broker 返回了重复 Agent")
        if set(assessment_ids) != expected_ids:
            raise ValueError("LLM Broker 必须完整评估所有候选 Agent")
        if not any(item.can_handle for item in decision.assessments):
            raise ValueError("LLM Broker 拒绝了全部候选 Agent")
        return {item.agent_id: item for item in decision.assessments}

    @staticmethod
    def _build_model(
        enabled: bool,
        model_config: ModelProfileConfig | None,
    ) -> Any | None:
        settings = get_settings()
        if not enabled or not settings.text_api_key:
            return None
        resolved = model_config or ModelProfileConfig(
            temperature=0.0,
            max_tokens=min(settings.llm_max_tokens, 900),
        )
        model = create_chat_model(
            temperature=resolved.temperature,
            max_tokens=resolved.max_tokens,
            enable_thinking=False,
        )
        return model.with_structured_output(BrokerRoutingDecision)

    def _fallback(
        self,
        capability: str,
        task: MeshTask,
        blackboard: MeshBlackboard,
        reason: str,
    ) -> tuple[_RegisteredAgent, float, list[dict[str, Any]]]:
        return self.registry.select_with_ranking(
            capability,
            task=task,
            blackboard=blackboard,
            routing_mode="metric_fallback",
            routing_fallback_reason=reason,
        )


REPLANNER_SYSTEM_PROMPT = """根据最终回复的审核问题规划一次修订。
只能append_tasks：response_revision（依赖已完成quality_review，内部修订评估并重写回复）
然后quality_recheck（依赖response_revision，审核新回复，且必须是唯一终止节点）。
每种能力只能出现一次；任务ID不得与completed_tasks重复。不要披露思维过程。
无法通过现有证据修正时可以finish且tasks=[]，系统会输出保守提示，不交付未通过的草稿。
"""


class ReplanSafetyValidator:
    """Validate an LLM-authored continuation before it reaches the executor."""

    allowed_capabilities = {"response_revision", "quality_recheck"}

    def __init__(self, registry: AgentRegistry, max_tasks: int = 6) -> None:
        self.registry = registry
        self.max_tasks = max(1, max_tasks)

    def to_plan(
        self,
        decision: ReplanDecision,
        *,
        completed_tasks: set[str],
    ) -> MeshExecutionPlan:
        if decision.action == "finish":
            if decision.tasks:
                raise ValueError("finish 决策不能携带新增任务")
            return MeshExecutionPlan(
                phase="completion",
                reason=decision.reason,
                tasks=[],
                planner="llm_replanner",
            )

        tasks = decision.tasks
        if not tasks:
            raise ValueError("append_tasks 决策必须包含任务")
        if len(tasks) > self.max_tasks:
            raise ValueError(f"动态计划最多允许 {self.max_tasks} 个任务")

        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("动态计划包含重复 task_id")
        duplicated = set(task_ids).intersection(completed_tasks)
        if duplicated:
            raise ValueError(f"动态计划复用了已完成 task_id: {sorted(duplicated)}")

        available = set(task_ids) | completed_tasks
        by_id = {task.task_id: task for task in tasks}
        for task in tasks:
            if task.capability not in self.allowed_capabilities:
                raise ValueError(f"动态计划使用了不安全能力: {task.capability}")
            if not self.registry.has_capability(task.capability):
                raise ValueError(f"动态计划能力未注册: {task.capability}")
            missing = set(task.depends_on) - available
            if missing:
                raise ValueError(
                    f"动态任务 {task.task_id} 依赖不存在: {sorted(missing)}"
                )
            if task.task_id in task.depends_on:
                raise ValueError(f"动态任务 {task.task_id} 不能依赖自身")

        self._reject_cycles(by_id)
        self._require_completed_root(tasks, by_id, completed_tasks)

        by_capability = {task.capability: task for task in tasks}
        if len(tasks) != 2 or set(by_capability) != self.allowed_capabilities:
            raise ValueError("修订计划必须且只能包含一次重写和一次复审")
        revision = by_capability["response_revision"]
        review = by_capability["quality_recheck"]
        if "quality_review" not in revision.depends_on:
            raise ValueError("回复修订必须依赖已完成的最终回复审核")
        if review.depends_on != [revision.task_id]:
            raise ValueError("复审必须直接依赖本轮回复修订")
        if any(review.task_id in task.depends_on for task in tasks):
            raise ValueError("复审必须是唯一终止节点")

        return MeshExecutionPlan(
            phase="llm_replan",
            reason=decision.reason,
            tasks=tasks,
            planner="llm_replanner",
        )

    def _reject_cycles(self, by_id: dict[str, MeshTask]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise ValueError("动态计划包含循环依赖")
            visiting.add(task_id)
            for dependency in by_id[task_id].depends_on:
                if dependency in by_id:
                    visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in by_id:
            visit(task_id)

    def _require_completed_root(
        self,
        tasks: list[MeshTask],
        by_id: dict[str, MeshTask],
        completed_tasks: set[str],
    ) -> None:
        def reaches_completed(task: MeshTask, seen: set[str]) -> bool:
            if set(task.depends_on).intersection(completed_tasks):
                return True
            for dependency in task.depends_on:
                if dependency in seen or dependency not in by_id:
                    continue
                if reaches_completed(by_id[dependency], seen | {dependency}):
                    return True
            return False

        unrooted = [
            task.task_id
            for task in tasks
            if not reaches_completed(task, {task.task_id})
        ]
        if unrooted:
            raise ValueError(
                f"动态任务必须依赖已有执行结果: {sorted(unrooted)}"
            )


class LLMReplanner:
    """Generate a structured continuation and fall back to deterministic rules."""

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        fallback_planner: AdaptivePlanBuilder,
        enabled: bool = True,
        max_tasks: int = 6,
        structured_model: Any | None = None,
        model_config: ModelProfileConfig | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.registry = registry
        self.fallback_planner = fallback_planner
        self.validator = ReplanSafetyValidator(registry, max_tasks=max_tasks)
        self.model = structured_model or self._build_model(enabled, model_config)
        self.timeout_seconds = max(0.1, timeout_seconds)

    async def continuation_plan(
        self,
        audit_result: AgentResult,
        blackboard: MeshBlackboard,
    ) -> MeshExecutionPlan:
        should_replan = bool(
            self.fallback_planner.max_replans > 0
            and audit_result.success
            and not getattr(audit_result, "passed", False)
            and getattr(audit_result, "retry_recommended", False)
        )
        if not should_replan:
            return self.fallback_planner.continuation_plan(audit_result)
        if self.model is None:
            return self._fallback(
                audit_result,
                "LLM Replanner 未启用或未配置模型凭证",
            )

        payload = {
            "request": {
                "scene": blackboard.request.scene,
                "query": blackboard.values.get("guide_query", ""),
                "budget": blackboard.request.context.get("budget"),
                "constraints": blackboard.request.context.get("constraints") or [],
                "shopping_goals": (
                    blackboard.request.context.get("shopping_goals") or []
                ),
            },
            "judge": {
                "passed": getattr(audit_result, "passed", False),
                "retry_recommended": getattr(
                    audit_result,
                    "retry_recommended",
                    False,
                ),
                "issues": self._issues(audit_result),
            },
            "completed_tasks": sorted(blackboard.completed_tasks),
            "available_blackboard_keys": sorted(blackboard.values),
            "allowed_capabilities": sorted(
                capability
                for capability in self.validator.allowed_capabilities
                if self.registry.has_capability(capability)
            ),
            "limits": {
                "max_tasks": self.validator.max_tasks,
                "remaining_replans": self.fallback_planner.max_replans,
            },
        }
        try:
            raw_decision = await asyncio.wait_for(
                self.model.ainvoke(
                    [
                        SystemMessage(content=REPLANNER_SYSTEM_PROMPT),
                        HumanMessage(
                            content=json.dumps(payload, ensure_ascii=False)
                        ),
                    ]
                ),
                timeout=self.timeout_seconds,
            )
            decision = (
                raw_decision
                if isinstance(raw_decision, ReplanDecision)
                else ReplanDecision.model_validate(raw_decision)
            )
            return self.validator.to_plan(
                decision,
                completed_tasks=blackboard.completed_tasks,
            )
        except Exception as exc:
            logger.warning("agent_mesh.replanner_fallback", error=str(exc))
            return self._fallback(audit_result, str(exc))

    def _build_model(
        self,
        enabled: bool,
        model_config: ModelProfileConfig | None,
    ) -> Any | None:
        settings = get_settings()
        if not enabled or not settings.text_api_key:
            return None
        resolved = model_config or ModelProfileConfig(
            temperature=0.0,
            max_tokens=min(settings.llm_max_tokens, 1200),
        )
        model = create_chat_model(
            temperature=resolved.temperature,
            max_tokens=resolved.max_tokens,
            enable_thinking=False,
        )
        return model.with_structured_output(ReplanDecision)

    def _fallback(
        self,
        audit_result: AgentResult,
        reason: str,
    ) -> MeshExecutionPlan:
        plan = self.fallback_planner.continuation_plan(audit_result)
        return plan.model_copy(
            update={
                "planner": "rule_fallback",
                "fallback_reason": reason,
            }
        )

    @staticmethod
    def _issues(audit_result: AgentResult) -> list[dict[str, Any]]:
        issues = getattr(audit_result, "issues", [])
        return [
            issue.model_dump() if hasattr(issue, "model_dump") else dict(issue)
            for issue in issues
            if hasattr(issue, "model_dump") or isinstance(issue, dict)
        ]


class MeshDAGExecutor:
    """Execute all dependency-ready tasks in parallel waves."""

    def __init__(
        self,
        registry: AgentRegistry,
        broker: HybridBroker | None = None,
    ) -> None:
        self.registry = registry
        self.broker = broker

    async def execute(
        self,
        plan: MeshExecutionPlan,
        blackboard: MeshBlackboard,
    ) -> None:
        self._validate_plan(plan, blackboard.completed_tasks)
        blackboard.add_plan(plan)
        pending = {task.task_id: task for task in plan.tasks}

        while pending:
            ready = [
                task
                for task in pending.values()
                if set(task.depends_on).issubset(blackboard.completed_tasks)
            ]
            if not ready:
                blocked = {key: value.depends_on for key, value in pending.items()}
                raise RuntimeError(f"任务 DAG 无可执行节点: {blocked}")

            blackboard.waves.append([task.task_id for task in ready])
            results = await asyncio.gather(
                *(self._execute_task(task, blackboard) for task in ready),
                return_exceptions=True,
            )
            for task, result in zip(ready, results, strict=True):
                pending.pop(task.task_id, None)
                if isinstance(result, BaseException):
                    raise RuntimeError(
                        f"Mesh 任务 {task.task_id} 执行失败: {result}"
                    ) from result
                blackboard.completed_tasks.add(task.task_id)

    async def _execute_task(
        self,
        task: MeshTask,
        blackboard: MeshBlackboard,
    ) -> None:
        if self.broker is None:
            selected, broker_score, broker_candidates = (
                self.registry.select_with_ranking(
                    task.capability,
                    task=task,
                    blackboard=blackboard,
                )
            )
        else:
            selected, broker_score, broker_candidates = (
                await self.broker.select_with_ranking(
                    task.capability,
                    task=task,
                    blackboard=blackboard,
                )
            )
        selected_candidate = next(
            candidate for candidate in broker_candidates if candidate["selected"]
        )
        trace = {
            "task_id": task.task_id,
            "capability": task.capability,
            "depends_on": task.depends_on,
            "agent_id": selected.card.agent_id,
            "broker_score": broker_score,
            "agent_bid": selected_candidate["bid"],
            "broker_candidates": broker_candidates,
            "broker_mode": selected_candidate["routing_mode"],
            "broker_fallback_reason": selected_candidate[
                "routing_fallback_reason"
            ],
            "broker_reason": (
                f"在 {len(broker_candidates)} 个候选 Agent 中选择综合评分最高者；"
                f"场景报价：{selected_candidate['bid']['reason']}"
            ),
            "status": "running",
            "latency_ms": 0.0,
            "output_keys": [],
            "error": None,
        }
        blackboard.task_traces.append(trace)
        started = time.perf_counter()
        success = False
        self.registry.mark_started(selected.card.agent_id)
        await report_progress(
            f"mesh:{task.task_id}",
            f"执行 {task.task_id}",
            "running",
            agent_id=selected.card.agent_id,
            depends_on=task.depends_on,
        )
        try:
            output = await selected.executor(task, blackboard)
            strategy = output.pop("_strategy", None)
            if strategy is not None:
                trace["strategy"] = strategy
            blackboard.publish(output)
            trace["status"] = "completed"
            trace["output_keys"] = sorted(output)
            success = True
        except Exception as exc:
            trace["status"] = "failed"
            trace["error"] = str(exc)
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            trace["latency_ms"] = round(latency_ms, 1)
            self.registry.mark_finished(
                selected.card.agent_id,
                success=success,
                latency_ms=latency_ms,
            )
            await report_progress(
                f"mesh:{task.task_id}",
                f"执行 {task.task_id}",
                trace["status"],
                agent_id=selected.card.agent_id,
                latency_ms=round(latency_ms, 1),
                depends_on=task.depends_on,
            )

    @staticmethod
    def _validate_plan(
        plan: MeshExecutionPlan,
        completed_tasks: set[str],
    ) -> None:
        ids = [task.task_id for task in plan.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("任务 DAG 包含重复 task_id")
        available = set(ids) | completed_tasks
        for task in plan.tasks:
            missing = set(task.depends_on) - available
            if missing:
                raise ValueError(
                    f"任务 {task.task_id} 依赖不存在的节点: {sorted(missing)}"
                )


class AdaptiveAgentMeshOrchestrator:
    """Dynamic DAG demo with broker selection, blackboard, judge and replan."""

    def __init__(
        self,
        *,
        workflow: SupervisorOrchestrator,
        ab_engine: ABTestEngine,
        sales_rag_service: SalesRagService | None,
        max_replans: int | None = None,
        llm_planner_enabled: bool | None = None,
        llm_broker_enabled: bool | None = None,
        llm_replanner_enabled: bool | None = None,
        max_plan_tasks: int | None = None,
        planner_model: Any | None = None,
        broker_model: Any | None = None,
        replanner_model: Any | None = None,
        agent_config: AgentSystemConfig | None = None,
        evaluation_store: AgentEvaluationStore | None = None,
    ) -> None:
        self.agent_config = agent_config or get_agent_system_config()
        orchestration = self.agent_config.orchestration
        resolved_max_replans = (
            orchestration.max_replans if max_replans is None else max_replans
        )
        resolved_replanner_enabled = (
            orchestration.llm_replanner_enabled
            if llm_replanner_enabled is None
            else llm_replanner_enabled
        )
        resolved_planner_enabled = (
            orchestration.llm_planner_enabled
            if llm_planner_enabled is None
            else llm_planner_enabled
        )
        resolved_broker_enabled = (
            orchestration.llm_broker_enabled
            if llm_broker_enabled is None
            else llm_broker_enabled
        )
        resolved_max_plan_tasks = (
            orchestration.max_plan_tasks
            if max_plan_tasks is None
            else max_plan_tasks
        )
        self.workflow = workflow
        self.ab_engine = ab_engine
        self.sales_rag_service = sales_rag_service
        self.evaluation_store = evaluation_store
        self.rule_planner = AdaptivePlanBuilder(max_replans=resolved_max_replans)
        self.registry = AgentRegistry(
            weights=self.agent_config.broker.weights.model_dump(),
            success_prior_calls=self.agent_config.broker.success_prior_calls,
            latency_ceiling_ms=self.agent_config.broker.latency_ceiling_ms,
        )
        self._register_agents()
        planner_definition = self.agent_config.agent("llm-planner")
        self.planner = LLMPlanner(
            registry=self.registry,
            fallback_planner=self.rule_planner,
            enabled=resolved_planner_enabled and planner_definition.enabled,
            max_tasks=resolved_max_plan_tasks,
            structured_model=planner_model,
            model_config=self.agent_config.resolved_model("llm-planner"),
            timeout_seconds=planner_definition.runtime.timeout_seconds,
        )
        broker_definition = self.agent_config.agent("llm-broker")
        self.broker = HybridBroker(
            registry=self.registry,
            enabled=resolved_broker_enabled and broker_definition.enabled,
            structured_model=broker_model,
            model_config=self.agent_config.resolved_model("llm-broker"),
            timeout_seconds=broker_definition.runtime.timeout_seconds,
        )
        replanner_definition = self.agent_config.agent("llm-replanner")
        self.replanner = LLMReplanner(
            registry=self.registry,
            fallback_planner=self.rule_planner,
            enabled=resolved_replanner_enabled and replanner_definition.enabled,
            max_tasks=resolved_max_plan_tasks,
            structured_model=replanner_model,
            model_config=self.agent_config.resolved_model("llm-replanner"),
            timeout_seconds=replanner_definition.runtime.timeout_seconds,
        )
        self.executor = MeshDAGExecutor(self.registry, broker=self.broker)

    async def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        experiment = self.ab_engine.assign(request.user_id)
        copy_experiment = self.ab_engine.assign(request.user_id, "copy_style")
        request = request.model_copy(
            update={
                "context": {
                    **request.context,
                    "experiment_config": {
                        **(experiment.get("config") or {}),
                        **(copy_experiment.get("config") or {}),
                    },
                }
            }
        )
        query = str(
            request.context.get("query")
            or request.context.get("keyword")
            or request.scene
            or ""
        ).strip()
        blackboard = MeshBlackboard(
            request=request,
            request_id=request_id,
            values={"guide_query": query},
        )

        logger.info(
            "agent_mesh.start",
            request_id=request_id,
            user_id=request.user_id,
        )
        try:
            await report_progress("mesh_planner", "生成动态执行计划", "running")
            initial_plan = await self.planner.initial_plan(
                request,
                include_knowledge=bool(
                    self.sales_rag_service is not None
                    and self.sales_rag_service.enabled
                    and query
                ),
            )
            await report_progress(
                "mesh_planner",
                "生成动态执行计划",
                "completed",
                task_count=len(initial_plan.tasks),
            )
            await self.executor.execute(initial_plan, blackboard)

            audit_result = blackboard.values["audit_result"]
            await report_progress("mesh_replanner", "检查是否需要重新规划", "running")
            continuation = await self.replanner.continuation_plan(
                audit_result,
                blackboard,
            )
            await report_progress(
                "mesh_replanner",
                "检查是否需要重新规划",
                "completed",
                task_count=len(continuation.tasks),
            )
            if continuation.phase in {"judge_replan", "llm_replan"}:
                blackboard.replan = {
                    "triggered": True,
                    "reason": continuation.reason,
                    "added_tasks": [task.task_id for task in continuation.tasks],
                    "planner": continuation.planner,
                    "fallback_reason": continuation.fallback_reason,
                }
            await self.executor.execute(continuation, blackboard)
            products, draft = self.workflow._reviewed_output(
                blackboard.values["final_products"], blackboard.values["guide_result"],
                blackboard.values["audit_result"],
            )
            blackboard.values.update({"final_products": products, "guide_result": draft})
            blackboard.values["product_rec_result"].data["candidate_evaluation"] = blackboard.values["evaluation_result"].data

            total_latency = (time.perf_counter() - started) * 1000
            final_audit = blackboard.values.get("audit_result")
            initial_judge_passed = blackboard.values.get("initial_judge_passed")
            final_judge_passed = self.workflow._audit_passed(final_audit)
            product_count = len(blackboard.values.get("final_products", []))
            business_success = bool(final_judge_passed and product_count)
            self._apply_business_feedback(blackboard, business_success)
            trace = blackboard.trace(self.registry)
            trace["agent_config_version"] = self.agent_config.version
            trace["total_latency_ms"] = round(total_latency, 1)
            trace["evaluation"] = {
                "initial_judge_passed": initial_judge_passed,
                "final_judge_passed": final_judge_passed,
                "business_success": business_success,
                "replan_recovered": bool(
                    blackboard.replan["triggered"]
                    and initial_judge_passed is False
                    and business_success
                ),
                "persisted": False,
            }
            trace["evaluation"]["persisted"] = self._persist_evaluation(
                blackboard=blackboard,
                trace=trace,
                initial_judge_passed=initial_judge_passed,
                final_judge_passed=final_judge_passed,
                business_success=business_success,
                product_count=product_count,
                issue_count=len(getattr(final_audit, "issues", [])),
                total_latency_ms=total_latency,
            )
            mesh_result = AgentResult(
                agent_name="agent_mesh",
                success=True,
                latency_ms=total_latency,
                data=trace,
                confidence=1.0,
            )
            response = RecommendationResponse(
                request_id=request_id,
                user_id=request.user_id,
                products=blackboard.values.get("final_products", []),
                marketing_copies=getattr(
                    blackboard.values.get("guide_result"),
                    "copies",
                    [],
                ),
                experiment_group=experiment.get("group", "control"),
                agent_results={
                    "product_rec": blackboard.values["product_rec_result"],
                    "quality_reviewer": blackboard.values["audit_result"],
                    "response_generation": blackboard.values["guide_result"],
                    "agent_mesh": mesh_result,
                },
                rag_trace=blackboard.values.get("rag_trace", {}),
                citations=blackboard.values.get("citations", []),
                orchestration_mode="adaptive_mesh",
                orchestration_trace=trace,
                total_latency_ms=total_latency,
            )
            logger.info(
                "agent_mesh.complete",
                request_id=request_id,
                wave_count=len(blackboard.waves),
                replanned=blackboard.replan["triggered"],
                total_latency_ms=round(total_latency, 1),
            )
            return response
        except Exception as exc:
            logger.exception(
                "agent_mesh.fallback",
                request_id=request_id,
                error=str(exc),
            )
            response = await self.workflow.recommend(request)
            total_latency = (time.perf_counter() - started) * 1000
            self._apply_business_feedback(blackboard, False)
            trace = blackboard.trace(self.registry)
            trace["agent_config_version"] = self.agent_config.version
            trace.update(
                {
                    "fallback": "supervisor_workflow",
                    "error": str(exc),
                    "total_latency_ms": round(total_latency, 1),
                    "evaluation": {
                        "initial_judge_passed": blackboard.values.get(
                            "initial_judge_passed"
                        ),
                        "final_judge_passed": None,
                        "business_success": False,
                        "replan_recovered": False,
                        "persisted": False,
                    },
                }
            )
            trace["evaluation"]["persisted"] = self._persist_evaluation(
                blackboard=blackboard,
                trace=trace,
                initial_judge_passed=blackboard.values.get(
                    "initial_judge_passed"
                ),
                final_judge_passed=None,
                business_success=False,
                product_count=0,
                issue_count=0,
                total_latency_ms=total_latency,
                fallback=True,
            )
            response.orchestration_mode = "workflow_fallback"
            response.orchestration_trace = trace
            response.agent_results["agent_mesh"] = AgentResult(
                agent_name="agent_mesh",
                success=False,
                latency_ms=total_latency,
                error=str(exc),
                data=trace,
                confidence=0.0,
            )
            return response

    def restore_persisted_metrics(self) -> int:
        """Restore Broker history after application startup creates DB tables."""
        if self.evaluation_store is None:
            return 0
        try:
            stats = self.evaluation_store.load_agent_runtime_stats()
            self.registry.restore_runtime(stats)
            return len(stats)
        except Exception as exc:
            logger.warning("agent_mesh.metrics_restore_failed", error=str(exc))
            return 0

    def _apply_business_feedback(
        self,
        blackboard: MeshBlackboard,
        business_success: bool,
    ) -> None:
        if blackboard.business_feedback_applied:
            return
        for task in blackboard.task_traces:
            credited = (
                task.get("status") == "completed"
                and task.get("capability") in QUALITY_CREDIT_CAPABILITIES
            )
            task["business_success"] = business_success if credited else None
            task["quality_feedback_source"] = (
                "final_response_quality_review" if credited else None
            )
            if credited:
                self.registry.mark_business_outcome(
                    str(task.get("agent_id") or ""),
                    success=business_success,
                )
        blackboard.business_feedback_applied = True

    def _persist_evaluation(
        self,
        *,
        blackboard: MeshBlackboard,
        trace: dict[str, Any],
        initial_judge_passed: bool | None,
        final_judge_passed: bool | None,
        business_success: bool,
        product_count: int,
        issue_count: int,
        total_latency_ms: float,
        fallback: bool = False,
    ) -> bool:
        if self.evaluation_store is None:
            return False
        try:
            return self.evaluation_store.record_mesh_run(
                request_id=blackboard.request_id,
                user_id=blackboard.request.user_id,
                scene=blackboard.request.scene,
                trace=trace,
                initial_judge_passed=initial_judge_passed,
                final_judge_passed=final_judge_passed,
                business_success=business_success,
                product_count=product_count,
                issue_count=issue_count,
                total_latency_ms=total_latency_ms,
                fallback=fallback,
            )
        except Exception as exc:
            logger.warning(
                "agent_mesh.evaluation_persist_failed",
                request_id=blackboard.request_id,
                error=str(exc),
            )
            return False

    def _register_agents(self) -> None:
        executor_registry: dict[str, TaskExecutor] = {
            "product_recommendation": self._product_recall,
            "knowledge_retrieval": self._knowledge_retrieval,
            "quality_reviewer": self._quality_review,
            "response_generation": self._response_synthesis,
        }
        self.agent_config.validate_mesh_executor_allowlist(
            set(executor_registry)
        )
        for agent_id, definition in self.agent_config.agents.items():
            if not definition.enabled or definition.broker is None:
                continue
            executor = executor_registry.get(definition.executor)
            if executor is None:
                raise ValueError(
                    f"Mesh Agent {agent_id} 的 executor 不在代码白名单: "
                    f"{definition.executor}"
                )
            broker = definition.broker
            self.registry.register(
                AgentCard(
                    agent_id=agent_id,
                    capabilities=definition.capabilities,
                    historical_success=broker.historical_success,
                    average_latency_ms=broker.average_latency_ms,
                    cost_level=broker.cost_level,
                    max_concurrency=definition.runtime.max_concurrency,
                ),
                executor,
            )

        required_capabilities = {
            "product_recall", "response_synthesis", "response_revision",
            "quality_review", "quality_recheck",
        }
        missing = sorted(
            capability
            for capability in required_capabilities
            if not self.registry.has_capability(capability)
        )
        if missing:
            raise ValueError(f"Agent 配置缺少 Mesh 必需能力: {missing}")

    @staticmethod
    def _is_complex_knowledge_query(query: str) -> bool:
        complex_terms = (
            "比较",
            "对比",
            "区别",
            "为什么",
            "原因",
            "综合",
            "分别",
            "依据",
            "推荐理由",
            "哪款",
            "优缺点",
            "权衡",
        )
        return len(query) > 45 or any(term in query for term in complex_terms)

    async def _product_recall(
        self,
        _task: MeshTask,
        blackboard: MeshBlackboard,
    ) -> dict[str, Any]:
        mode, reason = ProductRecAgent.select_recall_mode(blackboard.request.context)
        result = await self.workflow._recommend_products(
            blackboard.request,
            recall_mode=mode,
        )
        result.data["recall_mode"] = mode
        result.data["recall_reason"] = reason
        candidates = await self.workflow._filter_candidates(
            list(getattr(result, "products", [])), blackboard.request.context,
        )
        return {
            "product_rec_result": result,
            "candidate_products": candidates,
            "_strategy": {"mode": mode, "reason": reason},
        }

    async def _knowledge_retrieval(
        self,
        task: MeshTask,
        blackboard: MeshBlackboard,
    ) -> dict[str, Any]:
        if self.sales_rag_service is None:
            return {
                "active_context": dict(blackboard.request.context),
                "rag_trace": {},
                "citations": [],
            }
        products: list[Product] = blackboard.values.get("candidate_products", [])
        if not products:
            return self._empty_product_evidence_context(blackboard)
        requested_query = str(task.input.get("query") or "").strip()
        query = requested_query or blackboard.values["guide_query"]
        mode = "standard" if self._is_complex_knowledge_query(query) else "fast"
        retrieve = (
            self.sales_rag_service.augment
            if mode == "standard" else self.sales_rag_service.augment_fast
        )
        context = await retrieve(
            user_message=query,
            sales_context=blackboard.request.context,
            products=[product.model_dump() for product in products],
            memory=blackboard.request.context.get("session_memory") or {},
        )
        return {
            "active_context": context,
            "rag_trace": context.get("rag_trace") or {},
            "citations": context.get("citations") or [],
            "_strategy": {"mode": mode},
        }

    @staticmethod
    def _empty_product_evidence_context(
        blackboard: MeshBlackboard,
    ) -> dict[str, Any]:
        context = {
            **blackboard.request.context,
            "rag_context": [],
            "citations": [],
            "rag_trace": {
                "enabled": True,
                "applied": False,
                "reason": "no_candidate_products",
                "product_query_count": 0,
                "parallel_retrieval": False,
            },
        }
        return {
            "active_context": context,
            "rag_trace": context["rag_trace"],
            "citations": [],
        }

    async def _response_synthesis(
        self, task: MeshTask, blackboard: MeshBlackboard,
    ) -> dict[str, Any]:
        context = dict(blackboard.values.get("active_context") or blackboard.request.context)
        feedback = []
        if task.capability == "response_revision":
            feedback = list(blackboard.values.get("initial_audit_issues") or [])
            instruction = str(task.input.get("instruction") or task.input.get("focus") or "").strip()
            if instruction:
                feedback.append({"issue_type": "revision_instruction", "detail": instruction})
        request = blackboard.request.model_copy(update={"context": context})
        evaluation, draft, products = await self.workflow._draft_response(
            request, blackboard.values.get("candidate_products", []),
            blackboard.values["product_rec_result"], feedback=feedback,
        )
        return {
            "active_context": context, "evaluation_result": evaluation,
            "guide_result": draft, "final_products": products,
            "_strategy": {"mode": "revision" if feedback else "initial",
                          "reason": str(evaluation.data.get("evaluation_strategy", ""))},
        }

    async def _quality_review(
        self, task: MeshTask, blackboard: MeshBlackboard,
    ) -> dict[str, Any]:
        result = await self.workflow._review_response(
            blackboard.values["active_context"], blackboard.values["final_products"],
            blackboard.values["guide_result"],
        )
        revision = task.capability == "quality_recheck"
        result.data["revision_count"] = int(revision)
        values = {"audit_result": result}
        if not revision:
            values.update({
                "initial_judge_passed": self.workflow._audit_passed(result),
                "initial_audit_issues": self.workflow._audit_issues(result),
            })
        return values
