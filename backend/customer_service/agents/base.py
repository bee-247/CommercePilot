"""Shared customer-service Agent contracts and prompts."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CustomerServiceAgentSpec:
    name: str
    description: str
    system_prompt: str
    tools: list[Any]


BASE_PROMPT = (
    "你是智导（CommercePilot）系统中的专业客服 Agent。回答必须准确、克制、可执行，"
    "不得虚构商品参数、库存、优惠、物流时效或售后承诺。"
    "需要事实依据时使用知识库工具；证据不足时明确说明需要人工确认。"
    "不要泄露思维链。使用知识库内容时保留可用的来源信息。"
)

KNOWLEDGE_PROMPT = (
    "涉及商品说明、活动规则、物流、退换货、保修或平台政策时，"
    "优先调用 search_customer_knowledge。每轮最多检索一次，"
    "拿到结果后直接基于结果回答。"
)
