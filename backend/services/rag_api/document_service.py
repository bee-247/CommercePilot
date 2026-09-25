"""Knowledge-base document application service."""

from __future__ import annotations

import re
from pathlib import Path

from core.config import get_settings
from core.env import resolve_project_path
from fastapi import BackgroundTasks, HTTPException, UploadFile
from rag.ingestion.upload_jobs import (
    DELETE_STEPS,
    delete_job_manager,
    upload_job_manager,
)
from rag.storage.milvus_client import MilvusManager
from rag.storage.models import Resource, User
from repositories.rag_resource_repository import RagResourceRepository
from schemas.rag_schemas import (
    DocumentDeleteJobResponse,
    DocumentDeleteResponse,
    DocumentDeleteStartResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadJobResponse,
    DocumentUploadResponse,
    DocumentUploadStartResponse,
)
from sqlalchemy.orm import Session

from .document_delete_processor import DocumentDeleteProcessor
from .document_upload_processor import DocumentUploadProcessor

UPLOAD_DIR = resolve_project_path("data/documents")

resource_repository = RagResourceRepository()
upload_processor = DocumentUploadProcessor(
    resource_repository=resource_repository,
)
delete_processor = DocumentDeleteProcessor(
    resource_repository=resource_repository,
)
legacy_milvus_manager = MilvusManager()


def _is_supported_document(filename: str) -> bool:
    return filename.lower().endswith(
        (".pdf", ".docx", ".doc", ".xlsx", ".xls")
    )


async def _save_upload_file(file: UploadFile, file_path: Path) -> None:
    """Write uploads in chunks so large files are not buffered in memory."""
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    written = 0
    with file_path.open("wb") as destination:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"文件不能超过 {get_settings().max_upload_mb} MB",
                )
            destination.write(chunk)


def _normalize_resource_metadata(
    *,
    category: str = "",
    brand: str = "",
    business_line: str = "",
    document_type: str = "product",
    section_title: str = "",
    product_tags: str | list[str] | None = None,
) -> dict:
    if isinstance(product_tags, str):
        tags = [
            item.strip()
            for item in re.split(r"[,，、\n]", product_tags)
            if item.strip()
        ]
    elif isinstance(product_tags, list):
        tags = [
            str(item).strip()
            for item in product_tags
            if str(item).strip()
        ]
    else:
        tags = []
    return {
        "category": category.strip(),
        "brand": brand.strip(),
        "business_line": business_line.strip(),
        "document_type": document_type.strip() or "product",
        "section_title": section_title.strip(),
        "product_tags": tags,
    }


def _scoped_filename(filename: str, current_user: User) -> str:
    clean = Path(filename.strip()).name
    if current_user.role == "admin":
        return clean
    return f"user_{current_user.id}__{clean}"


def _display_filename(resource: Resource) -> str:
    return (resource.metadata_json or {}).get(
        "original_filename",
        resource.filename,
    )


def _upload_metadata(
    *,
    original_filename: str,
    current_user: User,
    category: str,
    brand: str,
    business_line: str,
    document_type: str,
    section_title: str,
    product_tags: str,
) -> tuple[dict, int | None, str]:
    metadata = _normalize_resource_metadata(
        category=category,
        brand=brand,
        business_line=business_line,
        document_type=document_type,
        section_title=section_title,
        product_tags=product_tags,
    )
    visibility = "public" if current_user.role == "admin" else "private"
    owner_id = None if current_user.role == "admin" else current_user.id
    metadata.update(
        {
            "original_filename": original_filename,
            "owner_id": owner_id,
            "visibility": visibility,
        }
    )
    return metadata, owner_id, visibility


async def list_documents(current_user: User) -> DocumentListResponse:
    """Return resources visible to the authenticated user."""
    try:
        resources = resource_repository.list_visible(current_user)
        documents = [
            DocumentInfo(
                resource_id=item.id,
                filename=item.filename,
                display_name=_display_filename(item),
                visibility=item.visibility,
                is_owner=item.owner_id == current_user.id,
                file_type=item.file_type,
                chunk_count=item.chunk_count,
                category=item.category,
                brand=item.brand,
                business_line=item.business_line,
                document_type=item.document_type,
                status=item.status,
                uploaded_at=item.created_at.isoformat(),
            )
            for item in resources
        ]
        if documents or current_user.role != "admin":
            return DocumentListResponse(documents=documents)
        return DocumentListResponse(documents=_legacy_documents())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"获取文档列表失败: {exc}",
        ) from exc


def _legacy_documents() -> list[DocumentInfo]:
    """Expose pre-resource-table Milvus documents to administrators."""
    legacy_milvus_manager.init_collection()
    results = legacy_milvus_manager.query(
        output_fields=["filename", "file_type"],
        limit=10000,
    )
    file_stats: dict[str, dict] = {}
    for item in results:
        filename = item.get("filename", "")
        if not filename:
            continue
        stats = file_stats.setdefault(
            filename,
            {
                "filename": filename,
                "file_type": item.get("file_type", ""),
                "chunk_count": 0,
                "status": "legacy",
            },
        )
        stats["chunk_count"] += 1
    return [DocumentInfo(**stats) for stats in file_stats.values()]


async def upload_document_async(
    *,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    category: str,
    brand: str,
    business_line: str,
    document_type: str,
    section_title: str,
    product_tags: str,
    current_user: User,
    db: Session,
) -> DocumentUploadStartResponse:
    original_filename = _validate_upload(file)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = _scoped_filename(original_filename, current_user)
    file_path = UPLOAD_DIR / filename
    metadata, owner_id, visibility = _upload_metadata(
        original_filename=original_filename,
        current_user=current_user,
        category=category,
        brand=brand,
        business_line=business_line,
        document_type=document_type,
        section_title=section_title,
        product_tags=product_tags,
    )
    resource = resource_repository.upsert(
        db,
        filename=filename,
        file_path=str(file_path),
        metadata=metadata,
        owner_id=owner_id,
        visibility=visibility,
        status="uploading",
    )
    job = upload_job_manager.create_job(filename)
    try:
        upload_job_manager.update_step(
            job["job_id"],
            "upload",
            1,
            "running",
            "正在保存文件到服务器",
        )
        await _save_upload_file(file, file_path)
        upload_job_manager.complete_step(
            job["job_id"],
            "upload",
            "文件已上传，等待后台处理",
        )
    except HTTPException as exc:
        file_path.unlink(missing_ok=True)
        resource_repository.update_status(resource.id, status="failed")
        upload_job_manager.fail_job(job["job_id"], "upload", str(exc.detail))
        raise
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        resource_repository.update_status(resource.id, status="failed")
        upload_job_manager.fail_job(
            job["job_id"],
            "upload",
            f"文件保存失败: {exc}",
        )
        raise HTTPException(
            status_code=500,
            detail=f"文件保存失败: {exc}",
        ) from exc

    resource_repository.update_status(resource.id, status="processing")
    background_tasks.add_task(
        upload_processor.process_job,
        job["job_id"],
        str(file_path),
        filename,
        resource.id,
        metadata,
    )
    return DocumentUploadStartResponse(
        job_id=job["job_id"],
        resource_id=resource.id,
        filename=filename,
        message="文件已上传，正在后台解析和向量化入库",
    )


async def get_upload_job(
    *,
    job_id: str,
    current_user: User,
) -> DocumentUploadJobResponse:
    job = upload_job_manager.get_job(job_id)
    if not job or not _can_view_job(job, current_user):
        raise HTTPException(status_code=404, detail="上传任务不存在或已过期")
    return DocumentUploadJobResponse(**job)


async def list_upload_jobs(
    *,
    current_user: User,
) -> list[DocumentUploadJobResponse]:
    jobs = sorted(
        upload_job_manager.list_jobs(),
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )
    return [
        DocumentUploadJobResponse(**job)
        for job in jobs
        if _can_view_job(job, current_user)
    ]


async def delete_document_async(
    *,
    filename: str,
    background_tasks: BackgroundTasks,
    current_user: User,
) -> DocumentDeleteStartResponse:
    resource_repository.require_modifiable(filename, current_user)
    job = delete_job_manager.create_job(
        filename,
        steps=DELETE_STEPS,
        current_step="prepare",
        message="等待删除",
        completion_step="parent_store",
    )
    delete_job_manager.update_step(
        job["job_id"],
        "prepare",
        1,
        "running",
        "删除任务已提交",
    )
    background_tasks.add_task(
        delete_processor.process_job,
        job["job_id"],
        filename,
    )
    return DocumentDeleteStartResponse(
        job_id=job["job_id"],
        filename=filename,
        message=f"正在删除 {filename}",
    )


async def get_delete_job(
    *,
    job_id: str,
    current_user: User,
) -> DocumentDeleteJobResponse:
    job = delete_job_manager.get_job(job_id)
    if not job or not _can_view_job(job, current_user):
        raise HTTPException(status_code=404, detail="删除任务不存在或已过期")
    return DocumentDeleteJobResponse(**job)


async def upload_document(
    *,
    file: UploadFile,
    category: str,
    brand: str,
    business_line: str,
    document_type: str,
    section_title: str,
    product_tags: str,
    current_user: User,
    db: Session,
) -> DocumentUploadResponse:
    original_filename = _validate_upload(file)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = _scoped_filename(original_filename, current_user)
    file_path = UPLOAD_DIR / filename
    metadata, owner_id, visibility = _upload_metadata(
        original_filename=original_filename,
        current_user=current_user,
        category=category,
        brand=brand,
        business_line=business_line,
        document_type=document_type,
        section_title=section_title,
        product_tags=product_tags,
    )
    resource = resource_repository.upsert(
        db,
        filename=filename,
        file_path=str(file_path),
        metadata=metadata,
        owner_id=owner_id,
        visibility=visibility,
        status="processing",
    )
    try:
        await _save_upload_file(file, file_path)
        parent_count, leaf_count = upload_processor.process_sync(
            file_path=str(file_path),
            filename=filename,
            resource_id=resource.id,
            metadata=metadata,
        )
    except HTTPException:
        file_path.unlink(missing_ok=True)
        resource_repository.update_status(resource.id, status="failed")
        raise
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        resource_repository.update_status(resource.id, status="failed")
        raise HTTPException(
            status_code=500,
            detail=f"文档处理失败: {exc}",
        ) from exc
    return DocumentUploadResponse(
        resource_id=resource.id,
        filename=filename,
        chunks_processed=leaf_count,
        message=(
            f"成功上传并处理 {filename}，叶子分块 {leaf_count} 个，"
            f"父级分块 {parent_count} 个"
        ),
    )


async def delete_document(
    *,
    filename: str,
    current_user: User,
) -> DocumentDeleteResponse:
    resource_repository.require_modifiable(filename, current_user)
    try:
        deleted_count = delete_processor.process_sync(filename)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"删除文档失败: {exc}",
        ) from exc
    return DocumentDeleteResponse(
        filename=filename,
        chunks_deleted=deleted_count,
        message=f"成功删除文档 {filename} 的索引数据（本地文件已保留）",
    )


def _validate_upload(file: UploadFile) -> str:
    original_filename = (file.filename or "").strip()
    if not original_filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not _is_supported_document(original_filename):
        raise HTTPException(
            status_code=400,
            detail="仅支持 PDF、Word 和 Excel 文档",
        )
    return Path(original_filename).name


def _can_view_job(job: dict, current_user: User) -> bool:
    if current_user.role == "admin":
        return True
    return str(job.get("filename", "")).startswith(
        f"user_{current_user.id}__"
    )
