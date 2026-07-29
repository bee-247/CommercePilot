from sqlalchemy.orm import declarative_base


Base = declarative_base()

from .session import (  # noqa: E402,F401
    ProductSessionLocal,
    SessionLocal,
    UserSessionLocal,
    init_db as _init_sales_db,
    product_engine,
    user_engine,
)
from rag.storage.database import init_db as _init_rag_db  # noqa: E402


def init_db() -> None:
    _init_sales_db()
    _init_rag_db()
