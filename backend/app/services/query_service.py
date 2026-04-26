from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json


class QueryService:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        self.override_path = self.project_root / "eval" / "query_type_overrides.json"

    def _load_overrides(self) -> dict[str, str]:
        if not self.override_path.exists():
            return {}
        payload = json.loads(self.override_path.read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in payload.get("overrides", {}).items()}

    def classify_query(self, query: str) -> tuple[str, str, str]:
        text = " ".join(query.lower().strip().split())
        overrides = self._load_overrides()
        if query in overrides:
            return overrides[query], overrides[query], "manual override matched exact query text"

        synonym_terms = ["also known as", "synonym", "abbreviation", "difference between", "compare"]
        treatment_terms = ["treatment", "therapy", "drug", "intervention", "manage", "prevention"]
        outcome_terms = ["effect", "outcome", "risk", "associated with", "impact", "adverse", "side effect"]
        broad_terms = ["how", "why", "mechanism", "pathway", "role of", "overview", "summary"]
        exact_terms = ["what is", "what are", "which gene", "which genes", "which protein", "identify", "define"]

        if any(term in text for term in synonym_terms):
            bucket = "synonym_heavy"
            rationale = "matched synonym/abbreviation comparison cues"
        elif any(term in text for term in treatment_terms):
            bucket = "treatment_intervention"
            rationale = "matched treatment/intervention vocabulary"
        elif any(term in text for term in outcome_terms):
            bucket = "outcome_effect"
            rationale = "matched outcome/effect/risk vocabulary"
        elif any(term in text for term in broad_terms):
            bucket = "broad_exploratory"
            rationale = "matched broad exploratory or mechanism-oriented wording"
        elif any(term in text for term in exact_terms) or len(text.split()) <= 6:
            bucket = "exact_terminology"
            rationale = "short, direct query with exact terminology emphasis"
        else:
            bucket = "broad_exploratory"
            rationale = "defaulted to broad exploratory because no stronger rule matched"
        return bucket, bucket, rationale


@lru_cache
def get_query_service() -> QueryService:
    return QueryService()

