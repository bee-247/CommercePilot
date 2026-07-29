"""LLM-assisted routing for customer-service requests."""

import os
from typing import Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field, field_validator

from core.env import load_project_env

load_project_env()

CustomerServiceRoute = Literal[
    "general",
    "faq_specialist",
    "sales_script_specialist",
    "quality_reviewer",
]


class ServiceSubTask(BaseModel):
    route: CustomerServiceRoute
    instruction: str = Field(min_length=1)

    @field_validator("instruction")
    @classmethod
    def _clean_instruction(cls, value: str) -> str:
        return value.strip()


class ServiceTaskPlan(BaseModel):
    tasks: list[ServiceSubTask] = Field(default_factory=list)


class RouteDecision(BaseModel):
    route: CustomerServiceRoute


_router_model = None


def _get_router_model():
    global _router_model
    if _router_model is None:
        api_key = os.getenv("ARK_API_KEY")
        model = os.getenv("MODEL")
        if not api_key or not model:
            return None
        _router_model = init_chat_model(
            model=model,
            model_provider="openai",
            api_key=api_key,
            base_url=os.getenv("BASE_URL"),
            temperature=0,
            stream_usage=True,
        )
    return _router_model


def route_customer_service_agent(user_text: str) -> CustomerServiceRoute:
    model = _get_router_model()
    if not model:
        return "general"
    prompt = f"""
你是智导（CommercePilot）系统的意图路由器。只选择一个 Agent：
- general：普通商品咨询、活动规则、物流、退换货、保修和混合问题；
- faq_specialist：生成或整理 FAQ、标准问答；
- sales_script_specialist：生成导购话术、异议处理、销售沟通脚本；
- quality_reviewer：审核或改写客服回复，识别事实与合规风险。

用户请求：
{user_text}
""".strip()
    try:
        return model.with_structured_output(RouteDecision).invoke(
            [{"role": "user", "content": prompt}]
        ).route
    except Exception:
        return "general"


def plan_customer_service_tasks(user_text: str) -> list[ServiceSubTask]:
    model = _get_router_model()
    if not model:
        return [
            ServiceSubTask(
                route=route_customer_service_agent(user_text),
                instruction=user_text,
            )
        ]
    prompt = f"""
你是导购客服任务规划器。将用户请求拆成 1 到 4 个顺序执行的子任务。
可选 route：general、faq_specialist、sales_script_specialist、quality_reviewer。
只有请求包含多个明确目标时才拆分；每条 instruction 必须能独立执行，
并保留商品类目、品牌、渠道、语气、政策和客户需求等约束。

用户请求：
{user_text}
""".strip()
    try:
        plan = model.with_structured_output(ServiceTaskPlan).invoke(
            [{"role": "user", "content": prompt}]
        )
        tasks = [task for task in plan.tasks[:4] if task.instruction]
        if tasks:
            return tasks
    except Exception:
        pass
    return [
        ServiceSubTask(
            route=route_customer_service_agent(user_text),
            instruction=user_text,
        )
    ]
