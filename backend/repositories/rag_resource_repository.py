"""Persistence boundary for RAG knowledge-base resources."""

from __future__ import annotations

from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session

from rag.storage.database import SessionLocal
from rag.storage.models import Resource, User


class RagResourceRepository:
    """Own Resource queries and mutations outside the API/service layer."""

    def list_visible(self, current_user: User) -> list[Resource]:
        db = SessionLocal()
        try:
            query = db.query(Resource)
            if current_user.role != "admin":
                query = query.filter(
                    (Resource.visibility == "public")
                    | (Resource.owner_id == current_user.id)
                )
            return query.order_by(Resource.updated_at.desc()).all()
        finally:
            db.close()

    def get_by_id(self, resource_id: int | None) -> Resource | None:
        if not resource_id:
            return None
        db = SessionLocal()
        try:
            return db.query(Resource).filter(Resource.id == resource_id).first()
        finally:
            db.close()

    def require_modifiable(self, filename: str, current_user: User) -> Resource:
        db = SessionLocal()
        try:
            resource = (
                db.query(Resource).filter(Resource.filename == filename).first()
            )
            if not resource:
                raise HTTPException(status_code=404, detail="文档不存在")
            if (
                current_user.role != "admin"
                and resource.owner_id != current_user.id
            ):
                raise HTTPException(
                    status_code=403,
                    detail="只能删除自己上传的个人资料",
                )
            return resource
        finally:
            db.close()

    def upsert(
        self,
        db: Session,
        *,
        filename: str,
        file_path: str,
        metadata: dict,
        owner_id: int | None,
        visibility: str,
        status: str,
        chunk_count: int = 0,
    ) -> Resource:
        resource = db.query(Resource).filter(Resource.filename == filename).first()
        payload = {
            "source_file": file_path,
            "owner_id": owner_id,
            "visibility": visibility,
            "file_type": self.detect_file_type(filename),
            "category": metadata.get("category", ""),
            "brand": metadata.get("brand", ""),
            "business_line": metadata.get("business_line", ""),
            "document_type": metadata.get("document_type", "product"),
            "status": status,
            "chunk_count": chunk_count,
            "updated_at": datetime.utcnow(),
            "metadata_json": {
                "original_filename": metadata.get("original_filename", filename),
                "section_title": metadata.get("section_title", ""),
                "product_tags": metadata.get("product_tags", []),
            },
        }
        if resource:
            for key, value in payload.items():
                setattr(resource, key, value)
        else:
            resource = Resource(filename=filename, **payload)
            db.add(resource)
        db.commit()
        db.refresh(resource)
        return resource

    def update_status(
        self,
        resource_id: int | None,
        *,
        status: str,
        chunk_count: int | None = None,
    ) -> None:
        if not resource_id:
            return
        db = SessionLocal()
        try:
            resource = (
                db.query(Resource).filter(Resource.id == resource_id).first()
            )
            if not resource:
                return
            resource.status = status
            resource.updated_at = datetime.utcnow()
            if chunk_count is not None:
                resource.chunk_count = chunk_count
            db.commit()
        finally:
            db.close()

    def delete_by_filename(self, filename: str) -> bool:
        db = SessionLocal()
        try:
            resource = (
                db.query(Resource).filter(Resource.filename == filename).first()
            )
            if not resource:
                return False
            db.delete(resource)
            db.commit()
            return True
        finally:
            db.close()

    @staticmethod
    def detect_file_type(filename: str) -> str:
        file_lower = filename.lower()
        if file_lower.endswith(".pdf"):
            return "PDF"
        if file_lower.endswith((".docx", ".doc")):
            return "Word"
        if file_lower.endswith((".xlsx", ".xls")):
            return "Excel"
        return ""
