"""Registry for routed customer-service Agents."""

from customer_service.agents import (
    faq_specialist,
    general,
    quality_reviewer,
    sales_script_specialist,
)
from customer_service.agents.base import CustomerServiceAgentSpec


def get_customer_service_agent_specs() -> dict[str, CustomerServiceAgentSpec]:
    specs = [
        general.build_spec(),
        faq_specialist.build_spec(),
        sales_script_specialist.build_spec(),
        quality_reviewer.build_spec(),
    ]
    return {spec.name: spec for spec in specs}
