from __future__ import annotations

import math

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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
from app.retrieval.utils import build_results, tokenize


class BM25Retriever(BaseRetriever):
    method_name = "bm25"

    def __init__(self, dataset_name: str, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__(dataset_name)
        self.k1 = k1
        self.b = b
        self.docs_df = None
        self.vectorizer = None
        self.term_matrix = None
        self.doc_lengths = None
        self.avg_doc_length = 0.0
        self.idf = None
        self._loaded = False

    def build(self) -> None:
        docs_df = load_dataset_docs(self.dataset_name)
        texts = docs_df["retrieval_text"].fillna("").astype(str).tolist()

        vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=True)
        term_matrix = vectorizer.fit_transform(texts).tocsc()

        doc_lengths = np.asarray(term_matrix.sum(axis=1)).ravel().astype(float)
        doc_freq = np.diff(term_matrix.indptr).astype(float)
        num_docs = term_matrix.shape[0]
        idf = np.array(
            [
                math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
                for df in doc_freq
            ],
            dtype=float,
        )
        avg_doc_length = float(doc_lengths.mean()) if num_docs else 0.0

        index_dir = get_method_index_dir(self.dataset_name, self.method_name)
        docs_df.to_parquet(index_dir / "docs.parquet", index=False)
        save_pickle(index_dir / "vectorizer.pkl", vectorizer)
        save_sparse_matrix(index_dir / "term_matrix.npz", term_matrix)
        np.save(index_dir / "doc_lengths.npy", doc_lengths)
        np.save(index_dir / "idf.npy", idf)
        write_json(
            index_dir / "meta.json",
            {
                "dataset_name": self.dataset_name,
                "method": self.method_name,
                "num_docs": num_docs,
                "avg_doc_length": avg_doc_length,
                "k1": self.k1,
                "b": self.b,
            },
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        index_dir = get_method_index_dir(self.dataset_name, self.method_name)
        meta_path = index_dir / "meta.json"
        if not meta_path.exists():
            self.build()

        meta = read_json(meta_path)
        self.docs_df = load_dataset_docs(self.dataset_name)
        self.vectorizer = load_pickle(index_dir / "vectorizer.pkl")
        self.term_matrix = load_sparse_matrix(index_dir / "term_matrix.npz").tocsc()
        self.doc_lengths = np.load(index_dir / "doc_lengths.npy")
        self.idf = np.load(index_dir / "idf.npy")
        self.avg_doc_length = float(meta["avg_doc_length"])
        self._loaded = True

    def search(self, query: str, top_k: int) -> list:
        self._ensure_loaded()

        query_counts = self.vectorizer.transform([query])
        query_indices = query_counts.indices
        query_data = query_counts.data
        if query_indices.size == 0:
            return []

        scores = np.zeros(self.term_matrix.shape[0], dtype=float)
        for token_index, query_tf in zip(query_indices, query_data, strict=False):
            column = self.term_matrix.getcol(token_index)
            if column.nnz == 0:
                continue
            rows = column.indices
            term_freqs = column.data.astype(float)
            numer = term_freqs * (self.k1 + 1.0)
            denom = term_freqs + self.k1 * (
                1.0 - self.b + self.b * (self.doc_lengths[rows] / max(self.avg_doc_length, 1.0))
            )
            scores[rows] += self.idf[token_index] * (numer / denom) * float(query_tf)

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
