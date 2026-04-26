from __future__ import annotations

import gc
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from app.clustering.types import ClusteringOutput
from app.retrieval.service import get_retrieval_service
from app.retrieval.storage import (
    DATA_ROOT,
    get_method_index_dir,
    load_dataset_docs,
    load_pickle,
    load_sparse_matrix,
)
from app.retrieval.types import SearchResult

LOGGER = logging.getLogger("bioseek.dense_clustering")


class ClusteringService:
    def __init__(self) -> None:
        self._docs_cache: dict[str, Any] = {}
        self._tfidf_matrix_cache: dict[str, Any] = {}
        self._dense_embeddings_cache: dict[str, np.ndarray] = {}
        self._tfidf_vectorizer_cache: dict[str, TfidfVectorizer] = {}
        self._pmid_to_index_cache: dict[str, dict[str, int]] = {}
        self._configure_dense_runtime()

    def _configure_dense_runtime(self) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            import torch

            torch.set_num_threads(1)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
        except Exception:
            pass

    def _get_docs_and_lookup(self, dataset_name: str):
        if dataset_name not in self._docs_cache:
            docs_df = load_dataset_docs(dataset_name)
            self._docs_cache[dataset_name] = docs_df
            self._pmid_to_index_cache[dataset_name] = {
                str(row["pmid"]): int(idx) for idx, row in docs_df.reset_index().iterrows()
            }
        return self._docs_cache[dataset_name], self._pmid_to_index_cache[dataset_name]

    def _load_tfidf_assets(self, dataset_name: str):
        if dataset_name not in self._tfidf_matrix_cache:
            index_dir = get_method_index_dir(dataset_name, "tfidf")
            self._tfidf_matrix_cache[dataset_name] = load_sparse_matrix(index_dir / "matrix.npz").tocsr()
            self._tfidf_vectorizer_cache[dataset_name] = load_pickle(index_dir / "vectorizer.pkl")
        return self._tfidf_matrix_cache[dataset_name], self._tfidf_vectorizer_cache[dataset_name]

    def _load_dense_embeddings(self, dataset_name: str) -> np.ndarray:
        if dataset_name not in self._dense_embeddings_cache:
            index_dir = get_method_index_dir(dataset_name, "dense")
            self._dense_embeddings_cache[dataset_name] = np.load(index_dir / "embeddings.npy")
        return self._dense_embeddings_cache[dataset_name]

    def _artifact_dir(self, query: str, dataset_name: str, method: str) -> Path:
        slug = (
            query.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("?", "")
            .replace(",", "")
        )
        path = DATA_ROOT / "processed" / "clusters" / "dense" / dataset_name / method / slug
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _vectorize_results(self, dataset_name: str, results: list[SearchResult], vector_space: str):
        _, pmid_lookup = self._get_docs_and_lookup(dataset_name)
        indices = [pmid_lookup[result.pmid] for result in results if result.pmid in pmid_lookup]
        if len(indices) != len(results):
            missing = [result.pmid for result in results if result.pmid not in pmid_lookup]
            raise ValueError(f"Some retrieved PMIDs were not found in dataset docs: {missing[:5]}")

        if vector_space == "tfidf":
            matrix, vectorizer = self._load_tfidf_assets(dataset_name)
            return matrix[indices], indices, vectorizer
        if vector_space == "dense":
            embeddings = self._load_dense_embeddings(dataset_name)
            return embeddings[indices], indices, None
        raise ValueError("vector_space must be either 'tfidf' or 'dense'")

    def _cluster_keywords(
        self,
        vector_space: str,
        cluster_vectors,
        cluster_results: list[SearchResult],
        shared_vectorizer: TfidfVectorizer | None,
        top_terms: int = 6,
    ) -> list[str]:
        if vector_space == "tfidf" and shared_vectorizer is not None:
            centroid = cluster_vectors.mean(axis=0)
            if hasattr(centroid, "A1"):
                centroid_values = centroid.A1
            else:
                centroid_values = np.asarray(centroid).ravel()
            feature_names = shared_vectorizer.get_feature_names_out()
            top_idx = np.argsort(centroid_values)[::-1][:top_terms]
            return [str(feature_names[idx]) for idx in top_idx if centroid_values[idx] > 0]

        local_vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        texts = [result.retrieval_text for result in cluster_results]
        if not texts:
            return []
        local_matrix = local_vectorizer.fit_transform(texts)
        centroid_values = np.asarray(local_matrix.mean(axis=0)).ravel()
        feature_names = local_vectorizer.get_feature_names_out()
        top_idx = np.argsort(centroid_values)[::-1][:top_terms]
        return [str(feature_names[idx]) for idx in top_idx if centroid_values[idx] > 0]

    def _representative_docs(
        self,
        vector_space: str,
        cluster_vectors,
        cluster_results: list[SearchResult],
        max_docs: int = 3,
    ) -> list[SearchResult]:
        if len(cluster_results) <= max_docs:
            return cluster_results

        if vector_space == "tfidf":
            centroid = cluster_vectors.mean(axis=0)
            scores = np.asarray(cluster_vectors @ centroid.T).ravel()
        else:
            centroid = np.asarray(cluster_vectors).mean(axis=0)
            scores = np.asarray(cluster_vectors) @ centroid

        top_idx = np.argsort(scores)[::-1][:max_docs]
        return [cluster_results[int(idx)] for idx in top_idx]

    def _result_to_dict(self, result: SearchResult) -> dict[str, Any]:
        return {
            "pmid": result.pmid,
            "score": result.score,
            "retrieval_text": result.retrieval_text,
            "metadata": result.metadata,
            "method": result.method,
            "dataset_name": result.dataset_name,
        }

    def _write_dense_checkpoint_files(
        self,
        artifact_dir: Path,
        results: list[SearchResult],
        embeddings: np.ndarray,
    ) -> dict[str, str]:
        LOGGER.info("Writing dense checkpoint embeddings to %s", artifact_dir)
        embeddings_path = artifact_dir / "dense_embeddings.npy"
        np.save(embeddings_path, embeddings)
        if not embeddings_path.exists():
            raise RuntimeError(f"Failed to write dense embeddings file: {embeddings_path}")
        LOGGER.info("Wrote dense embeddings: %s", embeddings_path)

        input_docs_path = artifact_dir / "dense_input_docs.parquet"
        docs_df = pd.DataFrame([self._result_to_dict(result) for result in results])
        docs_df.to_parquet(input_docs_path, index=False)
        if not input_docs_path.exists():
            raise RuntimeError(f"Failed to write dense input docs file: {input_docs_path}")
        LOGGER.info("Wrote dense input docs: %s", input_docs_path)

        return {
            "dense_embeddings": str(embeddings_path),
            "dense_input_docs": str(input_docs_path),
        }

    def _write_raw_cluster_assignments(
        self,
        artifact_dir: Path,
        results: list[SearchResult],
        labels: np.ndarray,
    ) -> str:
        assignments_path = artifact_dir / "raw_cluster_assignments.parquet"
        assignments_df = pd.DataFrame(
            {
                "pmid": [result.pmid for result in results],
                "cluster_id": labels.astype(int),
                "score": [float(result.score) for result in results],
            }
        )
        assignments_df.to_parquet(assignments_path, index=False)
        if not assignments_path.exists():
            raise RuntimeError(f"Failed to write cluster assignments file: {assignments_path}")
        LOGGER.info("Wrote raw cluster assignments: %s", assignments_path)
        return str(assignments_path)

    def _write_dense_summary(
        self,
        artifact_dir: Path,
        payload: dict[str, Any],
    ) -> str:
        summary_path = artifact_dir / "dense_clustering_summary.json"
        summary_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        if not summary_path.exists():
            raise RuntimeError(f"Failed to write dense clustering summary: {summary_path}")
        LOGGER.info("Wrote dense clustering summary: %s", summary_path)
        return str(summary_path)

    def diversify_by_cluster(self, grouped_results: dict[str, list[SearchResult]]) -> list[SearchResult]:
        diversified: list[SearchResult] = []
        buckets = {key: value[:] for key, value in grouped_results.items()}
        while True:
            added = False
            for cluster_id in sorted(buckets, key=lambda item: int(item)):
                if buckets[cluster_id]:
                    diversified.append(buckets[cluster_id].pop(0))
                    added = True
            if not added:
                break
        return diversified

    def run_dense_clustering(
        self,
        query: str,
        method: str,
        dataset_name: str,
        top_k: int,
        num_clusters: int,
    ) -> ClusteringOutput:
        artifact_dir = self._artifact_dir(query=query, dataset_name=dataset_name, method=method)
        artifact_paths: dict[str, str] = {}
        try:
            LOGGER.info("Dense clustering start: query=%s dataset=%s method=%s top_k=%s", query, dataset_name, method, top_k)
            retrieval_service = get_retrieval_service()
            LOGGER.info("Loading dense retrieval results")
            results = retrieval_service.search(
                query=query,
                top_k=top_k,
                dataset_name=dataset_name,
                method=method,
            )
            LOGGER.info("Dense retrieval returned %s documents", len(results))
            if len(results) < 2:
                raise ValueError("At least two dense retrieved documents are required for clustering.")

            LOGGER.info("Loading dense embeddings for selected results")
            vectors, _, _ = self._vectorize_results(dataset_name, results, "dense")
            embeddings = np.asarray(vectors, dtype=np.float32)
            LOGGER.info("Dense embeddings shape: %s", embeddings.shape)
            if embeddings.ndim != 2 or embeddings.shape[0] != len(results) or embeddings.shape[1] == 0:
                raise ValueError(f"Invalid dense embedding matrix shape: {embeddings.shape}")

            artifact_paths.update(
                self._write_dense_checkpoint_files(
                    artifact_dir=artifact_dir,
                    results=results,
                    embeddings=embeddings,
                )
            )

            actual_clusters = max(2, min(num_clusters, len(results)))
            LOGGER.info("Starting MiniBatchKMeans clustering with n_clusters=%s", actual_clusters)
            kmeans = MiniBatchKMeans(
                n_clusters=actual_clusters,
                random_state=42,
                batch_size=min(128, len(results)),
                n_init="auto",
            )
            labels = kmeans.fit_predict(embeddings)
            LOGGER.info("Clustering complete")

            artifact_paths["raw_cluster_assignments"] = self._write_raw_cluster_assignments(
                artifact_dir=artifact_dir,
                results=results,
                labels=labels,
            )

            silhouette = None
            if len(set(labels.tolist())) > 1 and len(results) > actual_clusters:
                LOGGER.info("Computing silhouette score")
                silhouette = float(silhouette_score(embeddings, labels))
                LOGGER.info("Silhouette score=%s", silhouette)

            grouped: dict[str, list[SearchResult]] = {}
            size_dist: dict[str, int] = {}
            term_summaries: dict[str, list[str]] = {}

            for cluster_id in range(actual_clusters):
                member_positions = [idx for idx, label in enumerate(labels.tolist()) if label == cluster_id]
                cluster_results = [results[idx] for idx in member_positions]
                cluster_vectors = embeddings[member_positions]

                keywords = self._cluster_keywords(
                    vector_space="dense",
                    cluster_vectors=cluster_vectors,
                    cluster_results=cluster_results,
                    shared_vectorizer=None,
                )
                representative_docs = self._representative_docs(
                    vector_space="dense",
                    cluster_vectors=cluster_vectors,
                    cluster_results=cluster_results,
                )
                representative_pmids = {doc.pmid for doc in representative_docs}
                ordered_cluster_results = representative_docs + [
                    doc for doc in cluster_results if doc.pmid not in representative_pmids
                ]
                grouped[str(cluster_id)] = ordered_cluster_results
                size_dist[str(cluster_id)] = len(cluster_results)
                term_summaries[str(cluster_id)] = keywords

            diversified = self.diversify_by_cluster(grouped)

            summary_payload = {
                "query": query,
                "dataset_name": dataset_name,
                "method": method,
                "vector_space": "dense",
                "num_clusters": actual_clusters,
                "status": "success",
                "error": None,
                "silhouette_score": silhouette,
                "cluster_size_distribution": size_dist,
                "representative_term_summaries": term_summaries,
            }
            artifact_paths["dense_summary"] = self._write_dense_summary(
                artifact_dir=artifact_dir,
                payload=summary_payload,
            )

            del embeddings
            gc.collect()

            return ClusteringOutput(
                query=query,
                dataset_name=dataset_name,
                method=method,
                vector_space="dense",
                num_clusters=actual_clusters,
                silhouette_score=silhouette,
                cluster_size_distribution=size_dist,
                representative_term_summaries=term_summaries,
                grouped_results=grouped,
                diversified_results=diversified,
                status="success",
                error=None,
                artifact_paths=artifact_paths,
            )
        except Exception as exc:
            LOGGER.exception("Dense clustering failed: %s", exc)
            failure_payload = {
                "query": query,
                "dataset_name": dataset_name,
                "method": method,
                "vector_space": "dense",
                "status": "failed",
                "error": str(exc),
                "artifact_paths": artifact_paths,
            }
            try:
                artifact_paths["dense_summary"] = self._write_dense_summary(
                    artifact_dir=artifact_dir,
                    payload=failure_payload,
                )
            except Exception:
                pass
            return ClusteringOutput(
                query=query,
                dataset_name=dataset_name,
                method=method,
                vector_space="dense",
                num_clusters=0,
                silhouette_score=None,
                cluster_size_distribution={},
                representative_term_summaries={},
                grouped_results={},
                diversified_results=[],
                status="failed",
                error=str(exc),
                artifact_paths=artifact_paths,
            )

    def cluster_search_results(
        self,
        query: str,
        method: str,
        dataset_name: str,
        top_k: int,
        num_clusters: int,
        vector_space: str,
    ) -> ClusteringOutput:
        if vector_space == "dense":
            return self.run_dense_clustering(
                query=query,
                method=method,
                dataset_name=dataset_name,
                top_k=top_k,
                num_clusters=num_clusters,
            )

        retrieval_service = get_retrieval_service()
        results = retrieval_service.search(
            query=query,
            top_k=top_k,
            dataset_name=dataset_name,
            method=method,
        )
        if len(results) < 2:
            raise ValueError("At least two retrieved documents are required for clustering.")

        vectors, _, shared_vectorizer = self._vectorize_results(dataset_name, results, vector_space)
        actual_clusters = max(2, min(num_clusters, len(results)))

        kmeans = MiniBatchKMeans(
            n_clusters=actual_clusters,
            random_state=42,
            batch_size=min(256, len(results)),
            n_init="auto",
        )
        labels = kmeans.fit_predict(vectors)

        silhouette = None
        if len(set(labels.tolist())) > 1 and len(results) > actual_clusters:
            silhouette = float(silhouette_score(vectors, labels))

        grouped: dict[str, list[SearchResult]] = {}
        size_dist: dict[str, int] = {}
        term_summaries: dict[str, list[str]] = {}

        for cluster_id in range(actual_clusters):
            member_positions = [idx for idx, label in enumerate(labels.tolist()) if label == cluster_id]
            cluster_results = [results[idx] for idx in member_positions]
            if vector_space == "tfidf":
                cluster_vectors = vectors[member_positions]
            else:
                cluster_vectors = np.asarray(vectors)[member_positions]

            keywords = self._cluster_keywords(
                vector_space=vector_space,
                cluster_vectors=cluster_vectors,
                cluster_results=cluster_results,
                shared_vectorizer=shared_vectorizer,
            )
            representative_docs = self._representative_docs(
                vector_space=vector_space,
                cluster_vectors=cluster_vectors,
                cluster_results=cluster_results,
            )
            representative_pmids = {doc.pmid for doc in representative_docs}
            ordered_cluster_results = representative_docs + [
                doc for doc in cluster_results if doc.pmid not in representative_pmids
            ]

            grouped[str(cluster_id)] = ordered_cluster_results
            size_dist[str(cluster_id)] = len(cluster_results)
            term_summaries[str(cluster_id)] = keywords

        diversified = self.diversify_by_cluster(grouped)

        return ClusteringOutput(
            query=query,
            dataset_name=dataset_name,
            method=method,
            vector_space=vector_space,
            num_clusters=actual_clusters,
            silhouette_score=silhouette,
            cluster_size_distribution=size_dist,
            representative_term_summaries=term_summaries,
            grouped_results=grouped,
            diversified_results=diversified,
            status="success",
            error=None,
            artifact_paths=None,
        )


@lru_cache
def get_clustering_service() -> ClusteringService:
    return ClusteringService()
