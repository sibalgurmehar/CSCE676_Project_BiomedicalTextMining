from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.retrieval.base import BaseRetriever
from app.retrieval.storage import (
    get_method_index_dir,
    load_dataset_docs,
    load_pickle,
    load_sparse_matrix,
    read_json,
    save_pickle,
    save_sparse_matrix,
    write_json,
)
from app.retrieval.utils import build_results


class TfidfRetriever(BaseRetriever):
    method_name = "tfidf"

    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name)
        self.docs_df = None
        self.vectorizer = None
        self.matrix = None
        self._loaded = False

    def build(self) -> None:
        docs_df = load_dataset_docs(self.dataset_name)
        texts = docs_df["retrieval_text"].fillna("").astype(str).tolist()

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100_000,
        )
        matrix = vectorizer.fit_transform(texts)

        index_dir = get_method_index_dir(self.dataset_name, self.method_name)
        docs_df.to_parquet(index_dir / "docs.parquet", index=False)
        save_pickle(index_dir / "vectorizer.pkl", vectorizer)
        save_sparse_matrix(index_dir / "matrix.npz", matrix)
        write_json(
            index_dir / "meta.json",
            {
                "dataset_name": self.dataset_name,
                "method": self.method_name,
                "num_docs": len(docs_df),
                "num_terms": len(vectorizer.vocabulary_),
            },
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        index_dir = get_method_index_dir(self.dataset_name, self.method_name)
        meta_path = index_dir / "meta.json"
        if not meta_path.exists():
            self.build()

        read_json(meta_path)
        self.docs_df = load_dataset_docs(self.dataset_name)
        self.vectorizer = load_pickle(index_dir / "vectorizer.pkl")
        self.matrix = load_sparse_matrix(index_dir / "matrix.npz").tocsr()
        self._loaded = True

    def search(self, query: str, top_k: int) -> list:
        self._ensure_loaded()
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        if scores.size == 0:
            return []

        top_k = min(top_k, scores.size)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        filtered = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        return build_results(
            docs_df=self.docs_df,
            indices=[idx for idx, _ in filtered],
            scores=[score for _, score in filtered],
            method=self.method_name,
            dataset_name=self.dataset_name,
        )

