from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import pandas as pd


class AnalysisService:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[3]

    def _read_json(self, relative_path: str) -> Any:
        path = self.project_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Analysis artifact not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_csv(self, relative_path: str) -> pd.DataFrame:
        path = self.project_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Analysis artifact not found: {path}")
        return pd.read_csv(path)

    def metrics_summary(self) -> dict[str, Any]:
        return {
            "dataset_name": "unified",
            "source_path": str(self.project_root / "eval/results/unified/summary.json"),
            "summary": self._read_json("eval/results/unified/summary.json"),
            "narrative": self._read_json("eval/results/unified/narrative.json"),
        }

    def diversity_summary(self) -> dict[str, Any]:
        payload = self._read_json("eval/results/unified/diversity/dense/diversity_metrics.json")
        return {
            "dataset_name": payload["dataset_name"],
            "source_path": str(self.project_root / "eval/results/unified/diversity/dense/diversity_metrics.json"),
            "summary": payload["method_level_diversity"],
            "narrative": payload["narrative"],
        }

    def query_type_summary(self) -> dict[str, Any]:
        payload = self._read_json("eval/results/unified/query_type_analysis/bucket_summary.json")
        return {
            "dataset_name": "unified",
            "source_path": str(self.project_root / "eval/results/unified/query_type_analysis/bucket_summary.json"),
            "summary": payload["buckets"],
            "narrative": payload["narrative"],
        }

    def best_worst_queries(self) -> dict[str, Any]:
        return {
            "dataset_name": "unified",
            "source_path": str(self.project_root / "eval/results/unified/best_worst_queries.json"),
            "payload": self._read_json("eval/results/unified/best_worst_queries.json"),
        }

    def query_metrics(self, query: str) -> dict[str, Any]:
        path = self.project_root / "eval/results/unified/per_query_metrics.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Analysis artifact not found: {path}")

        df = pd.read_parquet(path)
        normalized_query = " ".join(query.strip().lower().split())
        matched = df[df["query_text"].astype(str).str.lower().str.strip() == normalized_query].copy()

        if matched.empty:
            return {
                "dataset_name": "unified",
                "source_path": str(path),
                "query": query,
                "found": False,
                "payload": {
                    "message": "This query does not exactly match a BioASQ query in the unified evaluation set."
                },
            }

        matched = matched.sort_values("method")
        rows = matched[
            [
                "method",
                "query_id",
                "query_text",
                "query_type",
                "num_relevant_docs",
                "num_retrieved_docs",
                "precision_at_5",
                "precision_at_10",
                "recall_at_10",
                "mrr",
                "ndcg_at_5",
                "ndcg_at_10",
            ]
        ].to_dict(orient="records")

        return {
            "dataset_name": "unified",
            "source_path": str(path),
            "query": query,
            "found": True,
            "payload": {
                "query_id": str(matched.iloc[0]["query_id"]),
                "query_text": str(matched.iloc[0]["query_text"]),
                "query_type": str(matched.iloc[0]["query_type"]),
                "num_relevant_docs": int(matched.iloc[0]["num_relevant_docs"]),
                "rows": rows,
            },
        }


@lru_cache
def get_analysis_service() -> AnalysisService:
    return AnalysisService()
