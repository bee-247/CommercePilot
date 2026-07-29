from customer_service.agents.base import (
    BASE_PROMPT,
    KNOWLEDGE_PROMPT,
    CustomerServiceAgentSpec,
)
from customer_service.generation_tools import prepare_reply_review
from rag.utils.tools import search_customer_knowledge


def build_spec() -> CustomerServiceAgentSpec:
    return CustomerServiceAgentSpec(
        name="quality_reviewer",
        description="审核客服回复的准确性、合规性和服务质量。",
        system_prompt=(
            BASE_PROMPT
            + KNOWLEDGE_PROMPT
            + "从事实准确、需求覆盖、表达清晰、合规风险和下一步行动五方面"
            "审核客服回复，并给出可直接替换的改进版本。"
        ),
        tools=[search_customer_knowledge, prepare_reply_review],
    )
