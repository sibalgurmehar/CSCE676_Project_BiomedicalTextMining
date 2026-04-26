from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json

from app.config import get_app_settings


class DatasetService:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        self.settings = get_app_settings()

    def dataset_info(self) -> dict[str, object]:
        mapping_report_path = self.project_root / "data/processed/unified/mapping_report.json"
        pubmed_stats_path = self.project_root / "data/processed/pubmed_subset/stats.json"
        bioasq_stats_path = self.project_root / "data/processed/bioasq/stats.json"
        mapping_report = (
            json.loads(mapping_report_path.read_text(encoding="utf-8"))
            if mapping_report_path.exists()
            else {}
        )
        pubmed_stats = (
            json.loads(pubmed_stats_path.read_text(encoding="utf-8"))
            if pubmed_stats_path.exists()
            else {}
        )
        bioasq_stats = (
            json.loads(bioasq_stats_path.read_text(encoding="utf-8"))
            if bioasq_stats_path.exists()
            else {}
        )

        return {
            "project_name": self.settings.project.name,
            "retrieval_corpus": "The retrieval corpus is a curated PubMed subset with PMIDs.",
            "relevance_labels": "Relevance labels come from BioASQ Task B.",
            "join_strategy": "Exact PMID joining was used to align BioASQ relevance labels with PubMed documents.",
            "experimentation_scope": "The subset was chosen for practical local experimentation and student-friendly evaluation.",
            "unified_dataset_note": (
                "Unified evaluation uses BioASQ queries and relevance labels mapped onto the curated PubMed subset. "
                f"Current surviving-query count: {mapping_report.get('number_of_surviving_queries', 'unknown')}."
            ),
            "stats": {
                "pubmed_subset_docs": pubmed_stats.get("total_docs"),
                "bioasq_questions": bioasq_stats.get("num_questions"),
                "bioasq_unique_pmids": bioasq_stats.get("num_unique_pmids"),
                "surviving_queries": mapping_report.get("number_of_surviving_queries"),
                "mapped_relevant_docs": mapping_report.get("total_mapped_relevant_docs"),
                "pmid_coverage_percent": mapping_report.get("percentage_of_bioasq_relevant_pmids_found_in_corpus"),
            },
            "paths": {
                "pubmed_subset_docs": str(self.project_root / "data/processed/pubmed_subset/docs.parquet"),
                "unified_docs": str(self.project_root / "data/processed/unified/docs.parquet"),
                "unified_questions": str(self.project_root / "data/processed/unified/questions.parquet"),
                "unified_qrels": str(self.project_root / "data/processed/unified/qrels.parquet"),
                "mapping_report": str(mapping_report_path),
            },
        }


@lru_cache
def get_dataset_service() -> DatasetService:
    return DatasetService()
