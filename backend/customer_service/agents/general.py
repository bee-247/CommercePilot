from customer_service.agents.base import (
    BASE_PROMPT,
    KNOWLEDGE_PROMPT,
    CustomerServiceAgentSpec,
)
from rag.utils.tools import search_customer_knowledge


def build_spec() -> CustomerServiceAgentSpec:
    return CustomerServiceAgentSpec(
        name="general",
        description="处理普通咨询、商品知识、规则与售后问题。",
        system_prompt=BASE_PROMPT + KNOWLEDGE_PROMPT,
        tools=[search_customer_knowledge],
    )
