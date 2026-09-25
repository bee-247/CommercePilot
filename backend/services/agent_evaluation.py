"""Persistent evaluation events and aggregates for Adaptive Agent Mesh."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from database.models import AgentEvaluationRecord, MeshEvaluationRecord
from database.session import UserSessionLocal
from sqlalchemy.orm import Session

QUALITY_CREDIT_CAPABILITIES = frozenset(
    {
        "product_recall",
        "knowledge_retrieval",
        "response_synthesis",
        "response_revision",
        "inventory_filter",
        "candidate_evaluation",
        "candidate_revision",
    }
)

# Normalize old task records at read time; keep the original audit rows intact.
MERGED_AGENT_IDS = {
    "semantic-recall-agent": "product-recommendation",
    "profile-recall-agent": "product-recommendation",
    "popularity-recall-agent": "product-recommendation",
    "public-knowledge-specialist": "knowledge-retrieval",
    "fast-knowledge-retrieval-agent": "knowledge-retrieval",
    "semantic-candidate-evaluator": "candidate-evaluator",
    "lightweight-candidate-evaluator": "candidate-evaluator",
    "feedback-aware-reviser": "candidate-evaluator",
    "grounded-recommendation-judge": "quality-reviewer",
    "grounded-recommendation-judge-v2": "quality-reviewer",
    "recommendation-critic": "quality-reviewer",
    "evidence-aware-shopping-guide": "response-generation",
    "shopping-guide": "response-generation",
}

INTERNAL_STEP_IDS = {"candidate-evaluator", "deterministic-inventory-guard"}


class AgentEvaluationStore:
    """Append Mesh evaluations and rebuild runtime Agent statistics."""

    def __init__(
        self,
        session_factory: Callable[[], Session] = UserSessionLocal,
    ) -> None:
        self.session_factory = session_factory

    def record_mesh_run(
        self,
        *,
        request_id: str,
        user_id: str,
        scene: str,
        trace: dict[str, Any],
        initial_judge_passed: bool | None,
        final_judge_passed: bool | None,
        business_success: bool,
        product_count: int,
        issue_count: int,
        total_latency_ms: float,
        fallback: bool = False,
    ) -> bool:
        """Persist one idempotent request summary and its selected task events."""
        tasks = trace.get("tasks") or []
        replan = trace.get("replan") or {}
        replan_triggered = bool(replan.get("triggered"))
        replan_recovered = bool(
            replan_triggered
            and initial_judge_passed is False
            and business_success
        )

        with self.session_factory() as db:
            if db.get(MeshEvaluationRecord, request_id) is not None:
                return False
            db.add(
                MeshEvaluationRecord(
                    request_id=request_id,
                    user_id=user_id,
                    scene=scene,
                    initial_judge_passed=initial_judge_passed,
                    final_judge_passed=final_judge_passed,
                    business_success=business_success,
                    replan_triggered=replan_triggered,
                    replan_recovered=replan_recovered,
                    fallback=fallback,
                    total_latency_ms=total_latency_ms,
                    task_count=len(tasks),
                    product_count=product_count,
                    issue_count=issue_count,
                    trace_json=json.dumps(trace, ensure_ascii=False, default=str),
                )
            )
            for task in tasks:
                components = task.get("broker_candidates") or []
                selected = next(
                    (
                        candidate
                        for candidate in components
                        if candidate.get("selected")
                    ),
                    {},
                )
                scores = selected.get("components") or {}
                bid = task.get("agent_bid") or selected.get("bid") or {}
                capability = str(task.get("capability") or "")
                execution_success = task.get("status") == "completed"
                credited = (
                    execution_success
                    and capability in QUALITY_CREDIT_CAPABILITIES
                )
                db.add(
                    AgentEvaluationRecord(
                        request_id=request_id,
                        user_id=user_id,
                        scene=scene,
                        task_id=str(task.get("task_id") or ""),
                        capability=capability,
                        agent_id=str(task.get("agent_id") or ""),
                        execution_success=execution_success,
                        business_success=(business_success if credited else None),
                        latency_ms=float(task.get("latency_ms") or 0.0),
                        broker_score=float(task.get("broker_score") or 0.0),
                        historical_success_score=float(
                            scores.get("historical_success") or 0.0
                        ),
                        latency_score=float(scores.get("latency") or 0.0),
                        cost_score=float(scores.get("cost") or 0.0),
                        scenario_score=float(bid.get("scenario_score") or 0.0),
                        scenario_reason=str(bid.get("reason") or ""),
                        error=str(task.get("error") or ""),
                        replan_triggered=replan_triggered,
                    )
                )
            db.commit()
        return True

    def load_agent_runtime_stats(self) -> dict[str, dict[str, Any]]:
        """Rebuild Broker runtime state from append-only task evaluations."""
        with self.session_factory() as db:
            rows = (
                db.query(AgentEvaluationRecord)
                .order_by(AgentEvaluationRecord.id.asc())
                .all()
            )
        stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "business_feedback_count": 0,
                "business_successes": 0,
                "observed_latency_ms": None,
            }
        )
        for row in rows:
            item = stats[MERGED_AGENT_IDS.get(row.agent_id, row.agent_id)]
            item["calls"] += 1
            if row.execution_success:
                item["successes"] += 1
            else:
                item["failures"] += 1
            if row.business_success is not None:
                item["business_feedback_count"] += 1
                if row.business_success:
                    item["business_successes"] += 1
            previous = item["observed_latency_ms"]
            item["observed_latency_ms"] = (
                row.latency_ms
                if previous is None
                else previous * 0.8 + row.latency_ms * 0.2
            )
        return dict(stats)

    def get_summary(self, *, days: int = 30) -> dict[str, Any]:
        since = datetime.now(UTC).replace(tzinfo=None) - timedelta(
            days=max(1, min(days, 3650))
        )
        with self.session_factory() as db:
            requests = (
                db.query(MeshEvaluationRecord)
                .filter(MeshEvaluationRecord.created_at >= since)
                .all()
            )
            tasks = (
                db.query(AgentEvaluationRecord)
                .filter(AgentEvaluationRecord.created_at >= since)
                .all()
            )

        replanned = [item for item in requests if item.replan_triggered]
        recoverable = [
            item
            for item in requests
            if item.replan_triggered and item.initial_judge_passed is False
        ]
        judge_results = [
            item for item in requests if item.final_judge_passed is not None
        ]
        return {
            "period_days": max(1, min(days, 3650)),
            "overview": {
                "request_count": len(requests),
                "business_success_rate": self._rate(
                    sum(item.business_success for item in requests),
                    len(requests),
                ),
                "final_judge_pass_rate": self._rate(
                    sum(bool(item.final_judge_passed) for item in judge_results),
                    len(judge_results),
                ),
                "replan_rate": self._rate(len(replanned), len(requests)),
                "replan_recovery_rate": self._rate(
                    sum(item.replan_recovered for item in recoverable),
                    len(recoverable),
                ),
                "fallback_rate": self._rate(
                    sum(item.fallback for item in requests),
                    len(requests),
                ),
                "avg_total_latency_ms": self._average(
                    [item.total_latency_ms for item in requests]
                ),
            },
            "agents": self._group_task_metrics([
                item for item in tasks
                if MERGED_AGENT_IDS.get(item.agent_id, item.agent_id) not in INTERNAL_STEP_IDS
            ], "agent_id"),
            "historical_internal_steps": self._group_task_metrics([
                item for item in tasks
                if MERGED_AGENT_IDS.get(item.agent_id, item.agent_id) in INTERNAL_STEP_IDS
            ], "agent_id"),
            "capabilities": self._group_task_metrics(tasks, "capability"),
        }

    def _group_task_metrics(
        self,
        rows: list[AgentEvaluationRecord],
        field: str,
    ) -> list[dict[str, Any]]:
        groups: dict[str, list[AgentEvaluationRecord]] = defaultdict(list)
        for row in rows:
            name = str(getattr(row, field))
            if field == "agent_id":
                name = MERGED_AGENT_IDS.get(name, name)
            groups[name].append(row)

        result = []
        for name, items in sorted(groups.items()):
            feedback = [
                item for item in items if item.business_success is not None
            ]
            result.append(
                {
                    field: name,
                    "call_count": len(items),
                    "execution_success_rate": self._rate(
                        sum(item.execution_success for item in items),
                        len(items),
                    ),
                    "business_feedback_count": len(feedback),
                    "business_success_rate": self._rate(
                        sum(bool(item.business_success) for item in feedback),
                        len(feedback),
                    ),
                    "avg_latency_ms": self._average(
                        [item.latency_ms for item in items]
                    ),
                    "avg_broker_score": self._average(
                        [item.broker_score for item in items]
                    ),
                    "avg_scenario_score": self._average(
                        [item.scenario_score for item in items]
                    ),
                    "scenario_calibration_mae": self._average(
                        [
                            abs(
                                item.scenario_score
                                - float(bool(item.business_success))
                            )
                            for item in feedback
                        ]
                    ),
                }
            )
        return result

    @staticmethod
    def _rate(numerator: int | float, denominator: int) -> float:
        return round(float(numerator) / denominator, 4) if denominator else 0.0

    @staticmethod
    def _average(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0
