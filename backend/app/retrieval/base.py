from abc import ABC, abstractmethod

from app.retrieval.types import SearchResult


class BaseRetriever(ABC):
    method_name: str

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    @abstractmethod
    def build(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        raise NotImplementedError

    def search_batch(self, queries: list[str], top_k: int) -> list[list[SearchResult]]:
        return [self.search(query=query, top_k=top_k) for query in queries]

