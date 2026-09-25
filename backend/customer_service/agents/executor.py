"""One customer-service graph with request-scoped prompts and tool permissions."""

from dataclasses import dataclass

from agents.runtime import get_quality_reviewer
from core.agent_config import get_agent_system_config
from customer_service.task_profiles import TASK_PROFILES
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage, ToolMessage


@dataclass(frozen=True)
class CustomerServiceTaskContext:
    route: str = "general"


class CustomerServiceTaskMiddleware(AgentMiddleware):
    @staticmethod
    def _profile(request):
        context = request.runtime.context
        route = context.route if context is not None else "general"
        return TASK_PROFILES.get(route, TASK_PROFILES["general"])

    def _model_request(self, request):
        profile = self._profile(request)
        is_review = request.runtime.context is not None and request.runtime.context.route == "quality_reviewer"
        parameters = get_agent_system_config().model_profiles["structured_response"].model_dump()
        if is_review:
            parameters["temperature"] = 0.0
        return request.override(
            model=get_quality_reviewer().llm if is_review else request.model,
            system_message=SystemMessage(content=profile.system_prompt),
            tools=list(profile.tools),
            model_settings={**request.model_settings, **parameters},
        )

    def wrap_model_call(self, request, handler):
        return handler(self._model_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._model_request(request))

    def _tool_rejection(self, request):
        allowed = {tool.name for tool in self._profile(request).tools}
        if request.tool_call["name"] not in allowed:
            return ToolMessage(
                content="当前任务不允许调用该工具，请使用本任务提供的工具。",
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        return None

    def wrap_tool_call(self, request, handler):
        rejection = self._tool_rejection(request)
        return rejection if rejection is not None else handler(request)

    async def awrap_tool_call(self, request, handler):
        rejection = self._tool_rejection(request)
        return rejection if rejection is not None else await handler(request)


def build_customer_service_agent(model):
    tools = {
        tool.name: tool
        for profile in TASK_PROFILES.values()
        for tool in profile.tools
    }
    return create_agent(
        model=model,
        tools=list(tools.values()),
        middleware=[CustomerServiceTaskMiddleware()],
        context_schema=CustomerServiceTaskContext,
    )
