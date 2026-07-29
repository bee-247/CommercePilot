from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ProductBase(DeclarativeBase):
    pass


class UserBase(DeclarativeBase):
    pass


Base = ProductBase


class ProductRecord(ProductBase):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100), index=True, default="")
    price: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    description: Mapped[str] = mapped_column(Text, default="")
    brand: Mapped[str] = mapped_column(String(100), index=True, default="")
    seller_id: Mapped[str] = mapped_column(String(64), index=True, default="")
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    image_url: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), index=True, default="active")
    hot_score: Mapped[float] = mapped_column(Float, index=True, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class InventoryRecord(ProductBase):
    __tablename__ = "inventory"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    stock: Mapped[int] = mapped_column(Integer, default=0)
    reserved_stock: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class UserRecord(UserBase):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str] = mapped_column(String(32), default="")
    city: Mapped[str] = mapped_column(String(100), default="")
    segments_json: Mapped[str] = mapped_column(Text, default="[]")
    preferred_categories_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class UserBehaviorRecord(UserBase):
    __tablename__ = "user_behaviors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    product_id: Mapped[str] = mapped_column(String(64), index=True)
    behavior_type: Mapped[str] = mapped_column(String(32), index=True)
    scene: Mapped[str] = mapped_column(String(64), default="")
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )


class ConversationSessionRecord(UserBase):
    __tablename__ = "conversation_sessions"

    session_key: Mapped[str] = mapped_column(String(160), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(96), index=True)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[str] = mapped_column(Text, default="")
    last_query: Mapped[str] = mapped_column(Text, default="")
    last_image_query: Mapped[str] = mapped_column(Text, default="")
    last_image_summary: Mapped[str] = mapped_column(Text, default="")
    last_image_attributes_json: Mapped[str] = mapped_column(Text, default="[]")
    active_constraints_json: Mapped[str] = mapped_column(Text, default="[]")
    last_categories_json: Mapped[str] = mapped_column(Text, default="[]")
    last_keywords_json: Mapped[str] = mapped_column(Text, default="[]")
    last_recommended_product_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    messages_json: Mapped[str] = mapped_column(Text, default="[]")
    turns_json: Mapped[str] = mapped_column(Text, default="[]")
    long_term_candidates_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class UserLongTermMemoryRecord(UserBase):
    __tablename__ = "user_long_term_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(96), index=True)
    memory_type: Mapped[str] = mapped_column(String(64), index=True)
    memory_key: Mapped[str] = mapped_column(String(160), index=True, default="")
    state: Mapped[str] = mapped_column(String(32), index=True, default="active")
    value: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(64), default="session_end")
    observation_count: Mapped[int] = mapped_column(Integer, default=1)
    superseded_by_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    last_observed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class MemoryObservationRecord(UserBase):
    __tablename__ = "memory_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(96), index=True, default="")
    memory_key: Mapped[str] = mapped_column(String(160), index=True, default="")
    memory_type: Mapped[str] = mapped_column(String(64), index=True, default="other")
    value: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    evidence: Mapped[str] = mapped_column(Text, default="")
    scope: Mapped[str] = mapped_column(String(32), default="long_term")
    raw_text: Mapped[str] = mapped_column(Text, default="")
    existing_memory_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resolved_memory_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resolved_action: Mapped[str] = mapped_column(String(32), default="")
    resolution_reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
