from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from app.clustering.service import get_clustering_service
from app.retrieval.types import SearchResult
from app.retrieval.service import get_retrieval_service
from app.schemas.api import ClusterResponse, ClusterSummary, RetrievedDocument


class SearchService:
    def __init__(self) -> None:
        self.retrieval = get_retrieval_service()
        self.clustering = get_clustering_service()

    @staticmethod
    def _serialize_result(result: SearchResult) -> RetrievedDocument:
        return RetrievedDocument.model_validate(asdict(result))

    def search(self, query: str, method: str, dataset_name: str, top_k: int) -> list[RetrievedDocument]:
        results = self.retrieval.search(
            query=query,
            top_k=top_k,
            dataset_name=dataset_name,
            method=method,
        )
        return [self._serialize_result(result) for result in results]

    def compare(self, query: str, methods: list[str], dataset_name: str, top_k: int):
        output = []
        for method in methods:
            documents = self.search(
                query=query,
                method=method,
                dataset_name=dataset_name,
                top_k=top_k,
            )
            output.append(
                {
                    "method": method,
                    "dataset_name": dataset_name,
                    "documents": documents,
                }
            )
        return output

    def clusters(
        self,
        query: str,
        method: str,
        dataset_name: str,
        top_k: int,
        num_clusters: int,
        vector_space: str,
    ) -> ClusterResponse:
        output = self.clustering.cluster_search_results(
            query=query,
            method=method,
            dataset_name=dataset_name,
            top_k=top_k,
            num_clusters=num_clusters,
            vector_space=vector_space,
        )
        cluster_summaries = [
            ClusterSummary(
                cluster_id=int(cluster_id),
                cluster_size=int(output.cluster_size_distribution[cluster_id]),
                representative_keywords=output.representative_term_summaries[cluster_id],
                representative_docs=[
                    self._serialize_result(result)
                    for result in results[:3]
                ],
            )
            for cluster_id, results in sorted(output.grouped_results.items(), key=lambda item: int(item[0]))
        ]
        diversified_results = [
            self._serialize_result(result)
            for result in output.diversified_results
        ]
        return ClusterResponse(
            query=output.query,
            dataset_name=output.dataset_name,
            method=output.method,
            vector_space=output.vector_space,
            num_clusters=output.num_clusters,
            status=output.status,
            error=output.error,
            silhouette_score=output.silhouette_score,
            cluster_size_distribution=output.cluster_size_distribution,
            representative_term_summaries=output.representative_term_summaries,
            cluster_summaries=cluster_summaries,
            diversified_results=diversified_results,
            artifact_paths=output.artifact_paths,
        )


@lru_cache
def get_search_service() -> SearchService:
    return SearchService()
