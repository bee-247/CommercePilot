"""Shared role instances, independent of API and database startup."""

from functools import lru_cache


@lru_cache
def get_understanding_agent():
    from .conversation_understanding_agent import ConversationUnderstandingAgent

    return ConversationUnderstandingAgent()


@lru_cache
def get_response_agent():
    from .response_generation_agent import ResponseGenerationAgent

    return ResponseGenerationAgent()


@lru_cache
def get_quality_reviewer():
    from .quality_reviewer_agent import QualityReviewerAgent

    return QualityReviewerAgent()
