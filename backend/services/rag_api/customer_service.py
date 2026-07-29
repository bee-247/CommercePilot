"""Customer-service artifact and specialist workflow service."""

from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session

from customer_service.task_service import (
    generate_faq,
    generate_sales_script,
    review_service_reply,
)
from rag.storage.models import ServiceArtifact, User
from schemas.rag_schemas import (
    GenerateFaqRequest,
    GenerateSalesScriptRequest,
    ReviewServiceReplyRequest,
    ServiceArtifactCreate,
    ServiceArtifactInfo,
    ServiceArtifactListResponse,
    ServiceArtifactUpdate,
    ServiceTaskResponse,
)


class CustomerService:
    def list_artifacts(
        self,
        *,
        artifact_type: str,
        current_user: User,
        db: Session,
    ) -> ServiceArtifactListResponse:
        query = db.query(ServiceArtifact).filter(
            ServiceArtifact.owner_id == current_user.id
        )
        if artifact_type:
            query = query.filter(ServiceArtifact.artifact_type == artifact_type)
        rows = query.order_by(ServiceArtifact.updated_at.desc()).limit(100).all()
        return ServiceArtifactListResponse(
            artifacts=[self._artifact_info(row) for row in rows]
        )

    def create_artifact(
        self,
        *,
        request: ServiceArtifactCreate,
        current_user: User,
        db: Session,
    ) -> ServiceArtifactInfo:
        if not request.title.strip() or not request.artifact_type.strip():
            raise HTTPException(status_code=400, detail="材料标题和类型不能为空")
        artifact = ServiceArtifact(
            owner_id=current_user.id,
            artifact_type=request.artifact_type.strip(),
            title=request.title.strip(),
            prompt=request.prompt,
            content_json=request.content_json,
            source_chunk_ids=request.source_chunk_ids,
        )
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        return self._artifact_info(artifact)

    def update_artifact(
        self,
        *,
        artifact_id: int,
        request: ServiceArtifactUpdate,
        current_user: User,
        db: Session,
    ) -> ServiceArtifactInfo:
        artifact = self._owned(artifact_id, current_user, db)
        for key in (
            "artifact_type",
            "title",
            "prompt",
            "content_json",
            "source_chunk_ids",
        ):
            value = getattr(request, key)
            if value is not None:
                setattr(
                    artifact,
                    key,
                    value.strip() if isinstance(value, str) else value,
                )
        artifact.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(artifact)
        return self._artifact_info(artifact)

    def delete_artifact(
        self,
        *,
        artifact_id: int,
        current_user: User,
        db: Session,
    ) -> dict:
        artifact = self._owned(artifact_id, current_user, db)
        db.delete(artifact)
        db.commit()
        return {"id": artifact_id, "message": "客服材料已删除"}

    def generate_faq(
        self,
        *,
        request: GenerateFaqRequest,
        username: str,
    ) -> ServiceTaskResponse:
        if not request.topic.strip():
            raise HTTPException(status_code=400, detail="topic 不能为空")
        try:
            return ServiceTaskResponse(**generate_faq(request, username))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"FAQ 生成失败: {exc}") from exc

    def generate_script(
        self,
        *,
        request: GenerateSalesScriptRequest,
        username: str,
    ) -> ServiceTaskResponse:
        if not request.customer_need.strip():
            raise HTTPException(status_code=400, detail="customer_need 不能为空")
        try:
            return ServiceTaskResponse(**generate_sales_script(request, username))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"话术生成失败: {exc}") from exc

    def review_reply(
        self,
        *,
        request: ReviewServiceReplyRequest,
        username: str,
    ) -> ServiceTaskResponse:
        if not request.customer_message.strip() or not request.agent_reply.strip():
            raise HTTPException(
                status_code=400,
                detail="customer_message 和 agent_reply 不能为空",
            )
        try:
            return ServiceTaskResponse(**review_service_reply(request, username))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"回复质检失败: {exc}") from exc

    @staticmethod
    def _artifact_info(artifact: ServiceArtifact) -> ServiceArtifactInfo:
        return ServiceArtifactInfo(
            id=artifact.id,
            artifact_type=artifact.artifact_type,
            title=artifact.title,
            prompt=artifact.prompt,
            content_json=artifact.content_json,
            source_chunk_ids=artifact.source_chunk_ids,
            created_at=artifact.created_at.isoformat(),
            updated_at=artifact.updated_at.isoformat(),
        )

    @staticmethod
    def _owned(
        artifact_id: int,
        current_user: User,
        db: Session,
    ) -> ServiceArtifact:
        artifact = (
            db.query(ServiceArtifact)
            .filter(
                ServiceArtifact.id == artifact_id,
                ServiceArtifact.owner_id == current_user.id,
            )
            .first()
        )
        if not artifact:
            raise HTTPException(status_code=404, detail="客服材料不存在")
        return artifact
