"""Versioned, validated configuration for agents and mesh orchestration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from core.config import get_settings
from core.env import resolve_project_path
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OrchestrationConfig(BaseModel):
    mode: Literal["mesh", "workflow"] = "mesh"
    max_replans: int = Field(default=1, ge=0, le=5)
    max_plan_tasks: int = Field(default=6, ge=1, le=20)
    llm_planner_enabled: bool = True
    llm_broker_enabled: bool = True
    llm_replanner_enabled: bool = True


class BrokerWeightsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    historical_success: float = Field(ge=0.0, le=1.0)
    latency: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0, le=1.0)
    scenario: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> BrokerWeightsConfig:
        total = sum(self.model_dump().values())
        if abs(total - 1.0) > 0.0001:
            raise ValueError(f"Broker 权重之和必须为 1，当前为 {total:.4f}")
        return self


class BrokerConfig(BaseModel):
    weights: BrokerWeightsConfig
    success_prior_calls: float = Field(default=5.0, ge=0.0, le=1000.0)
    latency_ceiling_ms: float = Field(default=5000.0, gt=0.0)


class ModelProfileConfig(BaseModel):
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=700, ge=1, le=32768)


class AgentModelOverride(BaseModel):
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)


class AgentRuntimeConfig(BaseModel):
    timeout_seconds: float = Field(default=10.0, gt=0.0, le=300.0)
    max_attempts: int = Field(default=2, ge=1, le=10)
    max_concurrency: int = Field(default=4, ge=1, le=1000)


class AgentBrokerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    historical_success: float = Field(default=0.95, ge=0.0, le=1.0)
    average_latency_ms: float = Field(default=500.0, ge=0.0)
    cost_level: float = Field(default=1.0, ge=0.0, le=5.0)


class AgentDefinition(BaseModel):
    enabled: bool = True
    executor: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]{1,63}$")
    capabilities: list[str] = Field(default_factory=list)
    model_profile: str | None = None
    model: AgentModelOverride = Field(default_factory=AgentModelOverride)
    runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)
    broker: AgentBrokerConfig | None = None
    prompt_version: str = "v1"


class ServiceDefinition(BaseModel):
    """Internal service settings; services have no agent identity or capabilities."""

    model_profile: str
    runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)


class AgentSystemConfig(BaseModel):
    version: Literal[1] = 1
    orchestration: OrchestrationConfig
    broker: BrokerConfig
    model_profiles: dict[str, ModelProfileConfig]
    agents: dict[str, AgentDefinition]
    services: dict[str, ServiceDefinition] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> AgentSystemConfig:
        for name, service in self.services.items():
            if service.model_profile not in self.model_profiles:
                raise ValueError(f"Service {name} 引用了不存在的模型配置")
        for agent_id, definition in self.agents.items():
            if definition.model_profile not in {None, *self.model_profiles}:
                raise ValueError(
                    f"Agent {agent_id} 引用了不存在的模型配置: "
                    f"{definition.model_profile}"
                )
            if definition.broker is not None and not definition.capabilities:
                raise ValueError(f"Broker Agent {agent_id} 必须声明 capability")
        return self

    def agent(self, agent_id: str) -> AgentDefinition:
        try:
            return self.agents[agent_id]
        except KeyError as exc:
            raise KeyError(f"Agent 配置不存在: {agent_id}") from exc

    def resolved_model(self, agent_id: str) -> ModelProfileConfig:
        definition = self.agent(agent_id)
        profile = (
            self.model_profiles[definition.model_profile]
            if definition.model_profile
            else ModelProfileConfig()
        )
        updates = definition.model.model_dump(exclude_none=True)
        return profile.model_copy(update=updates)

    def validate_mesh_executor_allowlist(self, allowed: set[str]) -> None:
        invalid = {
            agent_id: definition.executor
            for agent_id, definition in self.agents.items()
            if definition.enabled
            and definition.broker is not None
            and definition.executor not in allowed
        }
        if invalid:
            raise ValueError(f"Mesh executor 不在代码白名单: {invalid}")


def load_agent_system_config(path: str | Path) -> AgentSystemConfig:
    config_path = resolve_project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Agent 配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Agent 配置文件必须是 YAML 对象")
    return AgentSystemConfig.model_validate(raw)


@lru_cache
def get_agent_system_config() -> AgentSystemConfig:
    return load_agent_system_config(get_settings().agent_config_file)
