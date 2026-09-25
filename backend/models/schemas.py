from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UserSegment(str, Enum):
    NEW_USER = "new_user"
    ACTIVE = "active"
    HIGH_VALUE = "high_value"
    PRICE_SENSITIVE = "price_sensitive"
    CHURN_RISK = "churn_risk"


class UserProfile(BaseModel):
    user_id: str
    age: int | None = None
    gender: str | None = None
    city: str | None = None
    segments: list[UserSegment] = Field(default_factory=list)
    preferred_categories: list[str] = Field(default_factory=list)
    price_range: tuple[float, float] = (0.0, 10000.0)
    recent_views: list[str] = Field(default_factory=list)
    recent_purchases: list[str] = Field(default_factory=list)
    rfm_score: dict[str, float] = Field(default_factory=dict)
    real_time_tags: dict[str, Any] = Field(default_factory=dict)


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    description: str = ""
    brand: str = ""
    seller_id: str = ""
    stock: int = 0
    tags: list[str] = Field(default_factory=list)
    score: float = 0.0
    image_url: str = ""


class RecommendationRequest(BaseModel):
    user_id: str
    scene: str = "homepage"
    num_items: int = 10
    context: dict[str, Any] = Field(default_factory=dict)


class ChatHistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user_id: str = "web_user"
    session_id: str | None = None
    message: str
    history: list[ChatHistoryMessage] = Field(default_factory=list)
    category: str | None = None
    num_items: int = 6


class AgentResult(BaseModel):
    agent_name: str
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


class ProductRecResult(AgentResult):
    agent_name: str = "product_rec"
    products: list[Product] = Field(default_factory=list)
    recall_strategy: str = ""


class ProductEvidence(BaseModel):
    field: str
    value: Any


class CandidateEvaluation(BaseModel):
    product_id: str
    fit_score: float = Field(default=0.0, ge=0.0, le=100.0)
    hard_constraints_passed: bool = True
    matched_preferences: list[str] = Field(default_factory=list)
    unmet_constraints: list[str] = Field(default_factory=list)
    unverified_requirements: list[str] = Field(default_factory=list)
    evidence: list[ProductEvidence] = Field(default_factory=list)
    reason: str = ""


class CandidateEvaluationResult(AgentResult):
    agent_name: str = "candidate_evaluation_step"
    evaluations: list[CandidateEvaluation] = Field(default_factory=list)


class QualityIssue(BaseModel):
    product_id: str = ""
    issue_type: str
    detail: str


class QualityReviewResult(AgentResult):
    agent_name: str = "quality_reviewer"
    passed: bool = True
    retry_recommended: bool = False
    issues: list[QualityIssue] = Field(default_factory=list)


class ShoppingGuideResult(AgentResult):
    agent_name: str = "response_generation"
    answer: str = ""
    copies: list[dict[str, str]] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    request_id: str
    user_id: str
    products: list[Product] = Field(default_factory=list)
    marketing_copies: list[dict[str, str]] = Field(default_factory=list)
    experiment_group: str = "control"
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    rag_trace: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    orchestration_mode: str = "workflow"
    orchestration_trace: dict[str, Any] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatResponse(BaseModel):
    answer: str
    recall_strategy: str = ""
    recall_reason: str = ""
    recommendation: RecommendationResponse
    token_usage: dict[str, int] = Field(default_factory=dict)
    rag_trace: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    request_latency_ms: float = 0.0
