from customer_service.agents.base import (
    BASE_PROMPT,
    KNOWLEDGE_PROMPT,
    CustomerServiceAgentSpec,
)
from customer_service.generation_tools import prepare_faq
from rag.utils.tools import search_customer_knowledge


def build_spec() -> CustomerServiceAgentSpec:
    return CustomerServiceAgentSpec(
        name="faq_specialist",
        description="生成和维护商品、售后及平台规则 FAQ。",
        system_prompt=(
            BASE_PROMPT
            + KNOWLEDGE_PROMPT
            + "将用户需求整理成问题清晰、答案简洁、带适用范围的客服 FAQ。"
        ),
        tools=[search_customer_knowledge, prepare_faq],
    )
