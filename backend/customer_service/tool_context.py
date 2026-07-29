"""Request-local identity for customer-service knowledge retrieval."""

from contextvars import ContextVar, Token


_SERVICE_USERNAME: ContextVar[str] = ContextVar(
    "service_username",
    default="",
)


def set_service_username(username: str) -> Token:
    return _SERVICE_USERNAME.set(str(username or ""))


def reset_service_username(token: Token) -> None:
    _SERVICE_USERNAME.reset(token)


def get_service_username() -> str:
    return _SERVICE_USERNAME.get()
