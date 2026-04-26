from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.retrieval.utils import tokenize
from app.services.search_service import get_search_service


class AIPolishService:
    def __init__(self) -> None:
        self.search_service = get_search_service()

    def _rewrite_suggestions(self, query: str) -> list[str]:
        lowered = query.lower().strip()
        suggestions: list[str] = []
        if "heart attack" in lowered:
            suggestions.append(lowered.replace("heart attack", "myocardial infarction"))
        if "drug adverse effects" in lowered:
            suggestions.append(lowered.replace("drug adverse effects", "medication safety and adverse events"))
        if "treatment" in lowered:
            suggestions.append(f"{lowered} clinical trial evidence")
        suggestions.append(f"{lowered} pubmed review")
        return list(dict.fromkeys(item.strip() for item in suggestions if item.strip()))[:3]

    def _follow_up_queries(self, query: str) -> list[str]:
        base = query.strip()
        candidates = [
            f"{base} biomarkers",
            f"{base} side effects",
            f"{base} randomized trial",
            f"{base} clinical outcomes",
        ]
        return candidates[:4]

    def _why_matches(self, query: str, method: str, dataset_name: str, top_k: int) -> list[dict[str, Any]]:
        documents = self.search_service.search(
            query=query,
            method=method,
            dataset_name=dataset_name,
            top_k=top_k,
        )
        query_tokens = set(tokenize(query))
        payload: list[dict[str, Any]] = []
        for document in documents[: min(top_k, 3)]:
            title_tokens = set(tokenize(document.metadata.get("title", "")))
            abstract_tokens = set(tokenize(document.metadata.get("abstract", "")))
            overlap = sorted((query_tokens & (title_tokens | abstract_tokens)))[:5]
            reasons: list[str] = []
            if overlap:
                reasons.append(f"shared query terms: {', '.join(overlap)}")
            mesh_terms = document.metadata.get("mesh_terms", [])
            if mesh_terms:
                reasons.append(f"matched biomedical theme near MeSH term '{mesh_terms[0]}'")
            journal = document.metadata.get("journal")
            if journal:
                reasons.append(f"appears in journal context '{journal}'")
            payload.append(
                {
                    "pmid": document.pmid,
                    "title": document.metadata.get("title", ""),
                    "explanation": "; ".join(reasons) or "retrieval score and semantic proximity supported this match",
                }
            )
        return payload

    def _refined_cluster_labels(self, query: str, method: str, dataset_name: str) -> list[dict[str, Any]]:
        cluster_response = self.search_service.clusters(
            query=query,
            method=method,
            dataset_name=dataset_name,
            top_k=20,
            num_clusters=5,
            vector_space="tfidf",
        )
        refined: list[dict[str, Any]] = []
        for cluster in cluster_response.cluster_summaries:
            keywords = cluster.representative_keywords[:3]
            label = " / ".join(word.title() for word in keywords) if keywords else f"Cluster {cluster.cluster_id}"
            refined.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "label": label,
                    "keywords": cluster.representative_keywords,
                }
            )
        return refined

    def polish(
        self,
        query: str,
        method: str | None,
        dataset_name: str | None,
        top_k: int,
        refine_clusters: bool,
        include_matches: bool,
    ) -> dict[str, Any]:
        resolved_method = method or "bm25"
        resolved_dataset = dataset_name or "pubmed_subset"
        return {
            "enabled": True,
            "ai_assisted": True,
            "mode": "heuristic_assist",
            "query": query,
            "dataset_name": resolved_dataset,
            "method": resolved_method,
            "rewrite_suggestions": self._rewrite_suggestions(query),
            "suggested_follow_up_queries": self._follow_up_queries(query),
            "why_this_matched": (
                self._why_matches(query, resolved_method, resolved_dataset, top_k)
                if include_matches
                else []
            ),
            "refined_cluster_labels": (
                self._refined_cluster_labels(query, resolved_method, resolved_dataset)
                if refine_clusters
                else []
            ),
            "note": (
                "AI-assisted polish is optional and does not affect retrieval evaluation. "
                "This environment is using a lightweight heuristic assist layer."
            ),
        }


@lru_cache
def get_ai_polish_service() -> AIPolishService:
    return AIPolishService()
