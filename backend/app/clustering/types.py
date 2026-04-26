from dataclasses import dataclass
from typing import Any

from app.retrieval.types import SearchResult


@dataclass(slots=True)
class ClusteredResult:
    cluster_id: int
    size: int
    representative_keywords: list[str]
    representative_docs: list[SearchResult]


@dataclass(slots=True)
class ClusteringOutput:
    query: str
    dataset_name: str
    method: str
    vector_space: str
    num_clusters: int
    silhouette_score: float | None
    cluster_size_distribution: dict[str, int]
    representative_term_summaries: dict[str, list[str]]
    grouped_results: dict[str, list[SearchResult]]
    diversified_results: list[SearchResult]
    status: str = "success"
    error: str | None = None
    artifact_paths: dict[str, str] | None = None
