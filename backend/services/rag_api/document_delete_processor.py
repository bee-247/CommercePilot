"""Document deletion processor."""

from __future__ import annotations

from repositories.rag_resource_repository import RagResourceRepository
from rag.ingestion.upload_jobs import delete_job_manager
from rag.retrieval.embedding import embedding_service
from rag.storage.milvus_client import MilvusManager
from rag.storage.parent_chunk_store import ParentChunkStore

class DocumentDeleteProcessor:
    """Delete all stored representations of one document."""

    def __init__(
        self,
        *,
        resource_repository: RagResourceRepository | None = None,
    ) -> None:
        self.resource_repository = (
            resource_repository or RagResourceRepository()
        )
        self.milvus_manager = MilvusManager()
        self.parent_chunk_store = ParentChunkStore()

    def process_job(self, job_id: str, filename: str) -> None:
        failed_step = "prepare"
        try:
            delete_job_manager.update_step(
                job_id,
                "prepare",
                20,
                "running",
                "正在初始化 Milvus 集合",
            )
            self.milvus_manager.init_collection()
            delete_job_manager.complete_step(
                job_id,
                "prepare",
                "删除任务已创建",
            )

            failed_step = "bm25"
            delete_job_manager.update_step(
                job_id,
                "bm25",
                20,
                "running",
                "正在同步 BM25 统计",
            )
            self._remove_bm25_stats(filename)
            delete_job_manager.complete_step(
                job_id,
                "bm25",
                "BM25 统计已同步",
            )

            failed_step = "milvus"
            delete_job_manager.update_step(
                job_id,
                "milvus",
                30,
                "running",
                "正在删除 Milvus 向量数据",
            )
            deleted_count = self._delete_vectors(filename)
            delete_job_manager.complete_step(
                job_id,
                "milvus",
                f"向量数据已删除：{deleted_count} 条",
            )

            failed_step = "parent_store"
            delete_job_manager.update_step(
                job_id,
                "parent_store",
                30,
                "running",
                "正在删除 PostgreSQL 父级分块",
            )
            self.parent_chunk_store.delete_by_filename(filename)
            self.resource_repository.delete_by_filename(filename)
            delete_job_manager.complete_step(
                job_id,
                "parent_store",
                "父级分块已删除",
            )

            delete_job_manager.complete_job(
                job_id,
                f"已删除 {filename}，向量数据 {deleted_count} 条",
            )
        except Exception as exc:
            delete_job_manager.fail_job(job_id, failed_step, str(exc))

    def process_sync(self, filename: str) -> int:
        self.milvus_manager.init_collection()
        self._remove_bm25_stats(filename)
        deleted_count = self._delete_vectors(filename)
        self.parent_chunk_store.delete_by_filename(filename)
        self.resource_repository.delete_by_filename(filename)
        return deleted_count

    def _remove_bm25_stats(self, filename: str) -> None:
        rows = self.milvus_manager.query_all(
            filter_expr=f'filename == "{filename}"',
            output_fields=["text"],
        )
        embedding_service.increment_remove_documents(
            [row.get("text") or "" for row in rows]
        )

    def _delete_vectors(self, filename: str) -> int:
        result = self.milvus_manager.delete(f'filename == "{filename}"')
        return result.get("delete_count", 0) if isinstance(result, dict) else 0
