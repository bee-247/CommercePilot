import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from core.env import load_project_env, resolve_sqlite_url

load_project_env()

DATABASE_URL = resolve_sqlite_url(
    os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/langchain_app",
    )
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def _ensure_column(table_name: str, column_name: str, column_sql: str) -> None:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns(table_name)}
    if column_name in columns:
        return
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"))


def _ensure_rag_schema_columns() -> None:
    """Add columns required by the unified customer-service RAG schema."""
    json_column_sql = "JSONB DEFAULT '[]'::jsonb NOT NULL" if engine.dialect.name == "postgresql" else "JSON DEFAULT '[]' NOT NULL"
    _ensure_column("resources", "owner_id", "INTEGER")
    _ensure_column("resources", "visibility", "VARCHAR(20) DEFAULT 'public' NOT NULL")
    _ensure_column("parent_chunks", "resource_id", "INTEGER")
    _ensure_column("parent_chunks", "owner_id", "INTEGER")
    _ensure_column("parent_chunks", "visibility", "VARCHAR(20) DEFAULT 'public' NOT NULL")
    _ensure_column("resources", "category", "VARCHAR(100) DEFAULT '' NOT NULL")
    _ensure_column("resources", "brand", "VARCHAR(100) DEFAULT '' NOT NULL")
    _ensure_column("resources", "business_line", "VARCHAR(100) DEFAULT '' NOT NULL")
    _ensure_column("resources", "document_type", "VARCHAR(50) DEFAULT 'product' NOT NULL")
    _ensure_column("parent_chunks", "category", "VARCHAR(100) DEFAULT '' NOT NULL")
    _ensure_column("parent_chunks", "brand", "VARCHAR(100) DEFAULT '' NOT NULL")
    _ensure_column("parent_chunks", "business_line", "VARCHAR(100) DEFAULT '' NOT NULL")
    _ensure_column("parent_chunks", "document_type", "VARCHAR(50) DEFAULT 'product' NOT NULL")
    _ensure_column("parent_chunks", "section_title", "VARCHAR(255) DEFAULT '' NOT NULL")
    _ensure_column("parent_chunks", "product_tags", json_column_sql)


def init_db() -> None:
    # Delayed import to avoid circular dependency.
    from . import models as _models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_rag_schema_columns()
