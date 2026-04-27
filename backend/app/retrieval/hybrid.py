from __future__ import annotations

import os

from app.retrieval.base import BaseRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.storage import load_dataset_docs
from app.retrieval.types import SearchResult
from app.retrieval.utils import doc_metadata, min_max_normalize


class HybridRetriever(BaseRetriever):
    method_name = "hybrid"

    def __init__(
        self,
        dataset_name: str,
        lexical_weight: float = 0.6,
        dense_weight: float = 0.4,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
    ) -> None:
        super().__init__(dataset_name)
        self.lexical_weight = float(
            os.getenv("HYBRID_LEXICAL_WEIGHT", lexical_weight)
        )
        self.dense_weight = float(
            os.getenv("HYBRID_DENSE_WEIGHT", dense_weight)
        )
        self.bm25 = bm25 or BM25Retriever(dataset_name)
        self.dense = dense or DenseRetriever(dataset_name)
        self.docs_df = load_dataset_docs(dataset_name)
        self.docs_by_pmid = {
            str(row["pmid"]): row for _, row in self.docs_df.iterrows()
        }

    def build(self) -> None:
        self.bm25.build()
        self.dense.build()

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        candidate_k = max(top_k * 5, 50)
        bm25_results = self.bm25.search(query=query, top_k=candidate_k)
        dense_results = self.dense.search(query=query, top_k=candidate_k)

        bm25_scores = {result.pmid: result.score for result in bm25_results}
        dense_scores = {result.pmid: result.score for result in dense_results}
        bm25_norm = min_max_normalize(bm25_scores)
        dense_norm = min_max_normalize(dense_scores)

        combined_pmids = sorted(set(bm25_scores) | set(dense_scores))
        if not combined_pmids:
            return []

        fused: list[SearchResult] = []
        for pmid in combined_pmids:
            row = self.docs_by_pmid[pmid]
            final_score = (
                self.lexical_weight * bm25_norm.get(pmid, 0.0)
                + self.dense_weight * dense_norm.get(pmid, 0.0)
            )
            fused.append(
                SearchResult(
                    pmid=pmid,
                    score=float(final_score),
                    retrieval_text=str(row["retrieval_text"]),
                    metadata=doc_metadata(row),
                    method=self.method_name,
                    dataset_name=self.dataset_name,
                )
            )

        fused.sort(key=lambda item: item.score, reverse=True)
        return fused[:top_k]
