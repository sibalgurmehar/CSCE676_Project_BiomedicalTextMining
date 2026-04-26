from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from app.retrieval.types import SearchResult

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def doc_metadata(row: pd.Series) -> dict[str, object]:
    mesh_terms = row.get("mesh_terms")
    if hasattr(mesh_terms, "tolist"):
        mesh_terms = mesh_terms.tolist()
    if mesh_terms is None:
        mesh_terms = []

    return {
        "title": str(row.get("title", "") or ""),
        "abstract": str(row.get("abstract", "") or ""),
        "journal": str(row.get("journal", "") or ""),
        "year": None if pd.isna(row.get("year")) else int(row["year"]),
        "mesh_terms": [str(term) for term in mesh_terms],
    }


def build_results(
    docs_df: pd.DataFrame,
    indices: Iterable[int],
    scores: Iterable[float],
    method: str,
    dataset_name: str,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for idx, score in zip(indices, scores, strict=False):
        row = docs_df.iloc[int(idx)]
        results.append(
            SearchResult(
                pmid=str(row["pmid"]),
                score=float(score),
                retrieval_text=str(row["retrieval_text"]),
                metadata=doc_metadata(row),
                method=method,
                dataset_name=dataset_name,
            )
        )
    return results


def min_max_normalize(scores_by_key: dict[str, float]) -> dict[str, float]:
    if not scores_by_key:
        return {}

    values = np.array(list(scores_by_key.values()), dtype=float)
    min_score = float(values.min())
    max_score = float(values.max())

    if max_score == min_score:
        return {key: 1.0 for key in scores_by_key}

    return {
        key: (float(score) - min_score) / (max_score - min_score)
        for key, score in scores_by_key.items()
    }

