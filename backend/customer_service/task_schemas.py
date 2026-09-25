"""Task contracts shared by understanding and customer-service execution."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

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
        value = value.strip()
        if not value:
            raise ValueError("客服任务不能为空")
        return value


class ServiceTaskPlan(BaseModel):
    tasks: list[ServiceSubTask] = Field(min_length=1, max_length=4)
