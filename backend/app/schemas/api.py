from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class RetrievedDocument(BaseModel):
    pmid: str
    score: float
    retrieval_text: str
    metadata: dict[str, Any]
    method: str
    dataset_name: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    method: str = Field(default="bm25")
    dataset_name: str = Field(default="pubmed_subset")
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResponse(BaseModel):
    query: str
    method: str
    dataset_name: str
    documents: list[RetrievedDocument]
    note: str


class MethodResults(BaseModel):
    method: str
    dataset_name: str
    documents: list[RetrievedDocument]


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=2)
    methods: list[str] = Field(default_factory=lambda: ["tfidf", "bm25", "dense", "hybrid"])
    dataset_name: str = Field(default="unified")
    top_k: int = Field(default=10, ge=1, le=100)


class CompareResponse(BaseModel):
    query: str
    dataset_name: str
    results: list[MethodResults]
    note: str


class ClusterSummary(BaseModel):
    cluster_id: int
    cluster_size: int
    representative_keywords: list[str]
    representative_docs: list[RetrievedDocument]


class ClusterRequest(BaseModel):
    query: str = Field(..., min_length=2)
    method: str = Field(default="bm25")
    dataset_name: str = Field(default="pubmed_subset")
    top_k: int = Field(default=30, ge=2, le=200)
    num_clusters: int = Field(default=5, ge=2, le=50)
    vector_space: str = Field(default="tfidf")


class ClusterResponse(BaseModel):
    query: str
    dataset_name: str
    method: str
    vector_space: str
    num_clusters: int
    status: str
    error: str | None
    silhouette_score: float | None
    cluster_size_distribution: dict[str, int]
    representative_term_summaries: dict[str, list[str]]
    cluster_summaries: list[ClusterSummary]
    diversified_results: list[RetrievedDocument]
    artifact_paths: dict[str, str] | None


class QueryAnalysisRequest(BaseModel):
    query: str = Field(..., min_length=2)


class QueryAnalysisResponse(BaseModel):
    query: str
    bucket_rule_based: str
    bucket_final: str
    rationale: str


class QueryMetricsRequest(BaseModel):
    query: str = Field(..., min_length=2)


class QueryMetricsResponse(BaseModel):
    dataset_name: str
    source_path: str
    query: str
    found: bool
    payload: dict[str, Any]


class SummaryResponse(BaseModel):
    dataset_name: str
    source_path: str
    summary: list[dict[str, Any]] | dict[str, Any]
    narrative: dict[str, Any] | None = None


class ArtifactResponse(BaseModel):
    dataset_name: str
    source_path: str
    payload: dict[str, Any]


class AIPolishRequest(BaseModel):
    query: str = Field(..., min_length=2)
    method: str | None = None
    dataset_name: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    refine_clusters: bool = Field(default=False)
    include_matches: bool = Field(default=True)


class AIPolishResponse(BaseModel):
    enabled: bool
    ai_assisted: bool
    mode: str
    query: str
    dataset_name: str | None
    method: str | None
    rewrite_suggestions: list[str]
    suggested_follow_up_queries: list[str]
    why_this_matched: list[dict[str, Any]]
    refined_cluster_labels: list[dict[str, Any]]
    note: str


class DatasetInfoResponse(BaseModel):
    project_name: str
    retrieval_corpus: str
    relevance_labels: str
    join_strategy: str
    experimentation_scope: str
    unified_dataset_note: str
    stats: dict[str, Any]
    paths: dict[str, str]
