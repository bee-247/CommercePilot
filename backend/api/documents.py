"""Knowledge-base document route group."""

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from core.auth import get_current_user, get_db
from rag.storage.models import User
from schemas.rag_schemas import (
    DocumentDeleteJobResponse,
    DocumentDeleteResponse,
    DocumentDeleteStartResponse,
    DocumentListResponse,
    DocumentUploadJobResponse,
    DocumentUploadResponse,
    DocumentUploadStartResponse,
)
from services.rag_api import document_service


router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(current_user: User = Depends(get_current_user)):
    return await document_service.list_documents(current_user=current_user)


@router.post("/documents/upload/async", response_model=DocumentUploadStartResponse)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form(""),
    brand: str = Form(""),
    business_line: str = Form(""),
    document_type: str = Form("product"),
    section_title: str = Form(""),
    product_tags: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return await document_service.upload_document_async(
        background_tasks=background_tasks,
        file=file,
        category=category,
        brand=brand,
        business_line=business_line,
        document_type=document_type,
        section_title=section_title,
        product_tags=product_tags,
        current_user=current_user,
        db=db,
    )


@router.get("/documents/upload/jobs/{job_id}", response_model=DocumentUploadJobResponse)
async def get_upload_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    return await document_service.get_upload_job(
        job_id=job_id,
        current_user=current_user,
    )


@router.get("/documents/upload/jobs", response_model=list[DocumentUploadJobResponse])
async def list_upload_jobs(current_user: User = Depends(get_current_user)):
    return await document_service.list_upload_jobs(current_user=current_user)


@router.delete("/documents/delete/async/{filename}", response_model=DocumentDeleteStartResponse)
async def delete_document_async(
    filename: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    return await document_service.delete_document_async(
        filename=filename,
        background_tasks=background_tasks,
        current_user=current_user,
    )


@router.get("/documents/delete/jobs/{job_id}", response_model=DocumentDeleteJobResponse)
async def get_delete_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    return await document_service.get_delete_job(
        job_id=job_id,
        current_user=current_user,
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(""),
    brand: str = Form(""),
    business_line: str = Form(""),
    document_type: str = Form("product"),
    section_title: str = Form(""),
    product_tags: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return await document_service.upload_document(
        file=file,
        category=category,
        brand=brand,
        business_line=business_line,
        document_type=document_type,
        section_title=section_title,
        product_tags=product_tags,
        current_user=current_user,
        db=db,
    )


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(
    filename: str,
    current_user: User = Depends(get_current_user),
):
    return await document_service.delete_document(
        filename=filename,
        current_user=current_user,
    )
