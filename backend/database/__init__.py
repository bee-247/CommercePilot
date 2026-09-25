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


def init_db() -> None:
    # Sales-only import scripts should not load the RAG PostgreSQL driver.
    # The full application still initializes both databases on startup.
    from rag.storage.database import init_db as _init_rag_db

    _init_sales_db()
    _init_rag_db()
