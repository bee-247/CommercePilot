"""Structured outputs for customer-service generation workflows."""

from typing import Literal

from pydantic import BaseModel, Field


class FaqItem(BaseModel):
    question: str
    answer: str
    applies_to: str = ""
    caveat: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)


class FaqSetOutput(BaseModel):
    items: list[FaqItem] = Field(min_length=1)
    limitation: str = ""


class SalesScriptOutput(BaseModel):
    opening: str
    need_confirmation: list[str] = Field(default_factory=list)
    recommendation: str
    objection_handling: list[str] = Field(default_factory=list)
    next_action: str
    compliance_notes: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    limitation: str = ""


class ReviewDimension(BaseModel):
    name: str
    score: float = Field(ge=0, le=20)
    comment: str


class ReplyReviewOutput(BaseModel):
    score: float = Field(ge=0, le=100)
    dimensions: list[ReviewDimension] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    rewritten_reply: str
    risk_level: Literal["low", "medium", "high"] = "medium"
    source_chunk_ids: list[str] = Field(default_factory=list)
    limitation: str = ""
