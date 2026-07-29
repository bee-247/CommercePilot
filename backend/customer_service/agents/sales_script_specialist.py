from customer_service.agents.base import (
    BASE_PROMPT,
    KNOWLEDGE_PROMPT,
    CustomerServiceAgentSpec,
)
from customer_service.generation_tools import prepare_sales_script
from rag.utils.tools import search_customer_knowledge


def build_spec() -> CustomerServiceAgentSpec:
    return CustomerServiceAgentSpec(
        name="sales_script_specialist",
        description="生成面向具体需求和渠道的导购客服话术。",
        system_prompt=(
            BASE_PROMPT
            + KNOWLEDGE_PROMPT
            + "输出自然的需求确认、推荐理由、异议处理和下一步行动话术，"
            "避免夸张、施压和无依据承诺。"
        ),
        tools=[search_customer_knowledge, prepare_sales_script],
    )
