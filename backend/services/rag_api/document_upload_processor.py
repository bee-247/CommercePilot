"""Document parsing and storage processor."""

from __future__ import annotations

from repositories.rag_resource_repository import RagResourceRepository
from rag.ingestion.document_loader import DocumentLoader
from rag.ingestion.upload_jobs import upload_job_manager
from rag.retrieval.embedding import embedding_service
from rag.storage.milvus_client import MilvusManager
from rag.storage.milvus_writer import MilvusWriter
from rag.storage.parent_chunk_store import ParentChunkStore

class DocumentUploadProcessor:
    """Process uploaded files independently from FastAPI route objects."""

    def __init__(
        self,
        *,
        resource_repository: RagResourceRepository | None = None,
    ) -> None:
        self.resource_repository = (
            resource_repository or RagResourceRepository()
        )
        self.loader = DocumentLoader()
        self.parent_chunk_store = ParentChunkStore()
        self.milvus_manager = MilvusManager()
        self.milvus_writer = MilvusWriter(
            embedding_service=embedding_service,
            milvus_manager=self.milvus_manager,
        )

    def process_job(
        self,
        job_id: str,
        file_path: str,
        filename: str,
        resource_id: int | None,
        metadata: dict,
    ) -> None:
        failed_step = "cleanup"
        try:
            upload_job_manager.complete_step(
                job_id,
                "upload",
                "文件已保存到服务器",
            )
            upload_job_manager.update_step(
                job_id,
                "cleanup",
                10,
                "running",
                "正在清理同名旧文档",
            )
            self.cleanup_existing(filename)
            upload_job_manager.complete_step(
                job_id,
                "cleanup",
                "旧版本清理完成",
            )

            failed_step = "parse"
            upload_job_manager.update_step(
                job_id,
                "parse",
                5,
                "running",
                "正在解析文档并执行三级分块",
            )
            parent_docs, leaf_docs, _ = self.parse(
                file_path=file_path,
                filename=filename,
                resource_id=resource_id,
                metadata=metadata,
            )
            upload_job_manager.complete_step(
                job_id,
                "parse",
                f"解析完成：父级分块 {len(parent_docs)} 个，叶子分块 {len(leaf_docs)} 个",
            )

            failed_step = "parent_store"
            upload_job_manager.update_step(
                job_id,
                "parent_store",
                20,
                "running",
                "正在写入父级分块",
            )
            self.parent_chunk_store.upsert_documents(parent_docs)
            upload_job_manager.complete_step(
                job_id,
                "parent_store",
                f"父级分块已入库：{len(parent_docs)} 个",
            )

            failed_step = "vector_store"
            total_leaf = len(leaf_docs)
            upload_job_manager.update_step(
                job_id,
                "vector_store",
                0,
                "running",
                f"正在向量化入库：0 / {total_leaf}",
                total_chunks=total_leaf,
                processed_chunks=0,
            )

            def on_progress(processed: int, total: int) -> None:
                percent = round(processed * 100 / total) if total else 100
                upload_job_manager.update_step(
                    job_id,
                    "vector_store",
                    percent,
                    "running",
                    f"正在向量化入库：{processed} / {total}",
                    total_chunks=total,
                    processed_chunks=processed,
                )

            self.milvus_writer.write_documents(
                leaf_docs,
                progress_callback=on_progress,
            )
            upload_job_manager.complete_step(
                job_id,
                "vector_store",
                f"向量化入库完成：{total_leaf} 个叶子分块",
            )

            self.resource_repository.update_status(
                resource_id,
                status="processed",
                chunk_count=total_leaf,
            )
            upload_job_manager.complete_job(
                job_id,
                f"成功上传并处理 {filename}",
            )
        except Exception as exc:
            self.resource_repository.update_status(
                resource_id,
                status="failed",
            )
            upload_job_manager.fail_job(job_id, failed_step, str(exc))

    def process_sync(
        self,
        *,
        file_path: str,
        filename: str,
        resource_id: int,
        metadata: dict,
    ) -> tuple[int, int]:
        self.cleanup_existing(filename)
        parent_docs, leaf_docs, _ = self.parse(
            file_path=file_path,
            filename=filename,
            resource_id=resource_id,
            metadata=metadata,
        )
        self.parent_chunk_store.upsert_documents(parent_docs)
        self.milvus_writer.write_documents(leaf_docs)
        self.resource_repository.update_status(
            resource_id,
            status="processed",
            chunk_count=len(leaf_docs),
        )
        return len(parent_docs), len(leaf_docs)

    def parse(
        self,
        *,
        file_path: str,
        filename: str,
        resource_id: int | None,
        metadata: dict,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        documents = self.loader.load_document(
            file_path,
            filename,
            metadata={**metadata, "resource_id": resource_id},
        )
        if not documents:
            raise ValueError("文档处理失败，未能提取内容")
        parent_docs = [
            doc
            for doc in documents
            if int(doc.get("chunk_level", 0) or 0) in (1, 2)
        ]
        leaf_docs = [
            doc
            for doc in documents
            if int(doc.get("chunk_level", 0) or 0) == 3
        ]
        if not leaf_docs:
            raise ValueError("文档处理失败，未生成可检索叶子分块")
        return parent_docs, leaf_docs, documents

    def cleanup_existing(self, filename: str) -> None:
        self.milvus_manager.init_collection()
        delete_expr = f'filename == "{filename}"'
        for cleanup in (
            lambda: self._remove_bm25_stats(filename),
            lambda: self.milvus_manager.delete(delete_expr),
            lambda: self.parent_chunk_store.delete_by_filename(filename),
            lambda: self.graph_sync_service.delete(filename),
        ):
            try:
                cleanup()
            except Exception:
                # Cleanup is best-effort so first-time uploads remain possible.
                continue

    def _remove_bm25_stats(self, filename: str) -> None:
        rows = self.milvus_manager.query_all(
            filter_expr=f'filename == "{filename}"',
            output_fields=["text"],
        )
        embedding_service.increment_remove_documents(
            [row.get("text") or "" for row in rows]
        )
