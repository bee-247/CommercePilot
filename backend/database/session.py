from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from core.config import get_settings
from core.env import resolve_sqlite_url

from .models import ProductBase, UserBase


settings = get_settings()
product_database_url = resolve_sqlite_url(
    settings.product_database_url or settings.database_url
)
user_database_url = resolve_sqlite_url(
    settings.user_database_url or settings.database_url
)

product_engine = create_engine(
    product_database_url,
    connect_args={"check_same_thread": False}
    if product_database_url.startswith("sqlite")
    else {},
    future=True,
)
user_engine = create_engine(
    user_database_url,
    connect_args={"check_same_thread": False}
    if user_database_url.startswith("sqlite")
    else {},
    future=True,
)

ProductSessionLocal = sessionmaker(
    bind=product_engine, autoflush=False, autocommit=False, future=True
)
UserSessionLocal = sessionmaker(
    bind=user_engine, autoflush=False, autocommit=False, future=True
)

SessionLocal = ProductSessionLocal


def init_db() -> None:
    """Create database tables only; this intentionally does not seed products."""
    ProductBase.metadata.create_all(bind=product_engine)
    UserBase.metadata.create_all(bind=user_engine)
    _upgrade_user_memory_schema()


def _upgrade_user_memory_schema() -> None:
    """Add memory lifecycle columns for existing SQLite/demo databases."""
    inspector = inspect(user_engine)
    if not inspector.has_table("user_long_term_memories"):
        return
    columns = {column["name"] for column in inspector.get_columns("user_long_term_memories")}
    statements = []
    if "memory_key" not in columns:
        statements.append("ALTER TABLE user_long_term_memories ADD COLUMN memory_key VARCHAR(160) DEFAULT ''")
    if "state" not in columns:
        statements.append("ALTER TABLE user_long_term_memories ADD COLUMN state VARCHAR(32) DEFAULT 'active'")
    if "observation_count" not in columns:
        statements.append("ALTER TABLE user_long_term_memories ADD COLUMN observation_count INTEGER DEFAULT 1")
    if "superseded_by_id" not in columns:
        statements.append("ALTER TABLE user_long_term_memories ADD COLUMN superseded_by_id INTEGER")
    if "updated_at" not in columns:
        statements.append("ALTER TABLE user_long_term_memories ADD COLUMN updated_at DATETIME")
    if "last_observed_at" not in columns:
        statements.append("ALTER TABLE user_long_term_memories ADD COLUMN last_observed_at DATETIME")
    if "last_used_at" not in columns:
        statements.append("ALTER TABLE user_long_term_memories ADD COLUMN last_used_at DATETIME")
    if "expires_at" not in columns:
        statements.append("ALTER TABLE user_long_term_memories ADD COLUMN expires_at DATETIME")
    if not statements:
        return
    with user_engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
