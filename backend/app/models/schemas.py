from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=5)
    method: str
    dataset_name: str = Field(default="unified")
    top_k: int = Field(default=3, ge=1, le=20)


class RetrievedDocument(BaseModel):
    pmid: str
    score: float
    retrieval_text: str
    metadata: dict
    method: str
    dataset_name: str


class SearchResponse(BaseModel):
    query: str
    method: str
    dataset_name: str
    documents: list[RetrievedDocument]
    note: str


class BatchSearchRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1)
    method: str
    dataset_name: str = Field(default="unified")
    top_k: int = Field(default=3, ge=1, le=20)


class BatchSearchResponse(BaseModel):
    method: str
    dataset_name: str
    results: list[SearchResponse]
    note: str
