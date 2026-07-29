"""Customer-service specialist API routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.auth import get_current_user, get_db
from rag.storage.models import User
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
from services.rag_api.customer_service import CustomerService


router = APIRouter(tags=["customer-service"])
customer_service = CustomerService()


@router.get(
    "/customer-service/artifacts",
    response_model=ServiceArtifactListResponse,
)
async def list_artifacts(
    artifact_type: str = "",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return customer_service.list_artifacts(
        artifact_type=artifact_type,
        current_user=current_user,
        db=db,
    )


@router.post("/customer-service/artifacts", response_model=ServiceArtifactInfo)
async def create_artifact(
    request: ServiceArtifactCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return customer_service.create_artifact(
        request=request,
        current_user=current_user,
        db=db,
    )


@router.patch(
    "/customer-service/artifacts/{artifact_id}",
    response_model=ServiceArtifactInfo,
)
async def update_artifact(
    artifact_id: int,
    request: ServiceArtifactUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return customer_service.update_artifact(
        artifact_id=artifact_id,
        request=request,
        current_user=current_user,
        db=db,
    )


@router.delete("/customer-service/artifacts/{artifact_id}")
async def delete_artifact(
    artifact_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return customer_service.delete_artifact(
        artifact_id=artifact_id,
        current_user=current_user,
        db=db,
    )


@router.post("/customer-service/faq/generate", response_model=ServiceTaskResponse)
async def generate_faq(
    request: GenerateFaqRequest,
    current_user: User = Depends(get_current_user),
):
    return customer_service.generate_faq(
        request=request,
        username=current_user.username,
    )


@router.post(
    "/customer-service/scripts/generate",
    response_model=ServiceTaskResponse,
)
async def generate_script(
    request: GenerateSalesScriptRequest,
    current_user: User = Depends(get_current_user),
):
    return customer_service.generate_script(
        request=request,
        username=current_user.username,
    )


@router.post(
    "/customer-service/replies/review",
    response_model=ServiceTaskResponse,
)
async def review_reply(
    request: ReviewServiceReplyRequest,
    current_user: User = Depends(get_current_user),
):
    return customer_service.review_reply(
        request=request,
        username=current_user.username,
    )
