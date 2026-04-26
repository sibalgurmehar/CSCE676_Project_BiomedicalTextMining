from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SearchResult:
    pmid: str
    score: float
    retrieval_text: str
    metadata: dict[str, Any]
    method: str
    dataset_name: str

