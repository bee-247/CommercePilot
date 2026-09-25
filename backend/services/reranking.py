"""Rerank via a user-provided local BGE directory or a compatible HTTP API."""

from __future__ import annotations

import math
import threading
from functools import lru_cache

import requests
from core.config import get_settings
from core.env import resolve_project_path
from core.model_clients import rerank_endpoint


class RerankingService:
    def __init__(self):
        self.settings = get_settings()
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(
            self.settings.rerank_model_path
            or (
                self.settings.rerank_model
                and self.settings.rerank_api_key
                and self.settings.rerank_base_url
            )
        )

    def _local_scores(self, query: str, documents: list[str]) -> list[float]:
        # No hub model ID is accepted as a download target. Both loaders are offline.
        path = resolve_project_path(self.settings.rerank_model_path)
        if not path.is_dir():
            raise ValueError("RERANK_MODEL_PATH 必须指向已下载的完整模型目录")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        with self._lock:
            if self._model is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(path), local_files_only=True, trust_remote_code=False
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(path), local_files_only=True, trust_remote_code=False
                )
                model.to(self.settings.rerank_device)
                model.eval()
                self._tokenizer, self._model = tokenizer, model
            scores = []
            with torch.inference_mode():
                for start in range(0, len(documents), self.settings.rerank_batch_size):
                    batch = documents[start : start + self.settings.rerank_batch_size]
                    inputs = self._tokenizer(
                        [(query, document) for document in batch],
                        padding=True,
                        truncation=True,
                        max_length=self.settings.rerank_max_length,
                        return_tensors="pt",
                    ).to(self.settings.rerank_device)
                    logits = self._model(**inputs, return_dict=True).logits
                    scores.extend(logits.view(-1).float().cpu().tolist())
            if len(scores) != len(documents) or not all(map(math.isfinite, scores)):
                raise ValueError("本地 Reranker 返回了无效分数")
            return scores

    def rerank(
        self, query: str, docs: list[dict], top_k: int
    ) -> tuple[list[dict], dict]:
        ranked = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
        local = bool(self.settings.rerank_model_path)
        metadata = {
            "rerank_enabled": self.enabled,
            "rerank_applied": False,
            "rerank_model": self.settings.rerank_model,
            "rerank_endpoint": "local" if local else rerank_endpoint(),
            "rerank_error": None,
            "candidate_count": len(ranked),
        }
        if not ranked or not self.enabled:
            return ranked[:top_k], metadata
        try:
            texts = [str(doc.get("text") or "") for doc in ranked]
            if local:
                scores = self._local_scores(query, texts)
                items = sorted(
                    (
                        {"index": i, "relevance_score": score}
                        for i, score in enumerate(scores)
                    ),
                    key=lambda item: item["relevance_score"],
                    reverse=True,
                )[:top_k]
            else:
                response = requests.post(
                    rerank_endpoint(),
                    headers={"Authorization": f"Bearer {self.settings.rerank_api_key}"},
                    json={
                        "model": self.settings.rerank_model,
                        "query": query,
                        "documents": texts,
                        "top_n": min(top_k, len(texts)),
                        "return_documents": False,
                    },
                    timeout=15,
                )
                response.raise_for_status()
                items = response.json()["results"]
            results, seen = [], set()
            for item in items:
                index = item["index"]
                score = float(item["relevance_score"])
                if (
                    type(index) is not int
                    or not 0 <= index < len(ranked)
                    or index in seen
                    or not math.isfinite(score)
                ):
                    raise ValueError("Reranker 返回了无效文档索引或分数")
                seen.add(index)
                results.append({**ranked[index], "rerank_score": score})
            if len(results) < min(top_k, len(ranked)):
                raise ValueError("Reranker 返回结果不完整")
            metadata["rerank_applied"] = True
            return results[:top_k], metadata
        except Exception as exc:
            # Keep retrieval usable when the local model is missing/incompatible.
            metadata["rerank_error"] = str(exc)
            return ranked[:top_k], metadata


@lru_cache
def get_reranking_service() -> RerankingService:
    return RerankingService()
