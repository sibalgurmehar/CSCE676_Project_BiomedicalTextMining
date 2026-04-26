from __future__ import annotations

from functools import lru_cache

from app.retrieval.base import BaseRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.tfidf import TfidfRetriever
from app.retrieval.types import SearchResult


class RetrievalService:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], BaseRetriever] = {}

    def _create_retriever(self, dataset_name: str, method: str) -> BaseRetriever:
        normalized = method.lower()
        if normalized == "tfidf":
            return TfidfRetriever(dataset_name)
        if normalized == "bm25":
            return BM25Retriever(dataset_name)
        if normalized == "dense":
            return DenseRetriever(dataset_name)
        if normalized == "hybrid":
            return HybridRetriever(dataset_name)
        raise ValueError(f"Unsupported retrieval method '{method}'.")

    def get_retriever(self, dataset_name: str, method: str) -> BaseRetriever:
        key = (dataset_name, method.lower())
        if key not in self._cache:
            self._cache[key] = self._create_retriever(dataset_name, method)
        return self._cache[key]

    def search(self, query: str, top_k: int, dataset_name: str, method: str) -> list[SearchResult]:
        retriever = self.get_retriever(dataset_name=dataset_name, method=method)
        return retriever.search(query=query, top_k=top_k)

    def search_batch(
        self,
        queries: list[str],
        top_k: int,
        dataset_name: str,
        method: str,
    ) -> list[list[SearchResult]]:
        retriever = self.get_retriever(dataset_name=dataset_name, method=method)
        return retriever.search_batch(queries=queries, top_k=top_k)

    def build_indexes(self, dataset_name: str, methods: list[str]) -> None:
        for method in methods:
            retriever = self.get_retriever(dataset_name=dataset_name, method=method)
            retriever.build()


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService()

