"""Task profiles for one customer-service executor; profiles are not agents."""

from dataclasses import dataclass
from typing import Any

from customer_service.generation_tools import (
    prepare_faq,
    prepare_reply_review,
    prepare_sales_script,
)
from rag.utils.tools import search_customer_knowledge


@dataclass(frozen=True)
class CustomerServiceTaskProfile:
    system_prompt: str
    tools: tuple[Any, ...]


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


TASK_PROFILES = {
    "general": CustomerServiceTaskProfile(
        system_prompt=BASE_PROMPT + KNOWLEDGE_PROMPT,
        tools=(search_customer_knowledge,),
    ),
    "faq_specialist": CustomerServiceTaskProfile(
        system_prompt=BASE_PROMPT + KNOWLEDGE_PROMPT
        + "将用户需求整理成问题清晰、答案简洁、带适用范围的客服 FAQ。",
        tools=(search_customer_knowledge, prepare_faq),
    ),
    "sales_script_specialist": CustomerServiceTaskProfile(
        system_prompt=BASE_PROMPT + KNOWLEDGE_PROMPT
        + "输出自然的需求确认、推荐理由、异议处理和下一步行动话术，"
        "避免夸张、施压和无依据承诺。",
        tools=(search_customer_knowledge, prepare_sales_script),
    ),
    "quality_reviewer": CustomerServiceTaskProfile(
        system_prompt=BASE_PROMPT + KNOWLEDGE_PROMPT
        + "从事实准确、需求覆盖、表达清晰、合规风险和下一步行动五方面"
        "审核客服回复，并给出可直接替换的改进版本。",
        tools=(search_customer_knowledge, prepare_reply_review),
    ),
}
