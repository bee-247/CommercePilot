"""Request-scoped progress events for streamed Agent execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from typing import Any


ProgressReporter = Callable[[dict[str, Any]], Awaitable[None]]
_progress_reporter: ContextVar[ProgressReporter | None] = ContextVar(
    "execution_progress_reporter",
    default=None,
)


def set_progress_reporter(reporter: ProgressReporter) -> Token:
    return _progress_reporter.set(reporter)


def reset_progress_reporter(token: Token) -> None:
    _progress_reporter.reset(token)


async def report_progress(
    stage: str,
    label: str,
    status: str,
    **details: Any,
) -> None:
    reporter = _progress_reporter.get()
    if reporter is None:
        return
    await reporter(
        {
            "type": "progress",
            "stage": stage,
            "label": label,
            "status": status,
            **details,
        }
    )
