from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    BatchSearchRequest,
    BatchSearchResponse,
    SearchRequest,
    SearchResponse,
)
from app.retrieval.service import get_retrieval_service

router = APIRouter()

@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    try:
        documents = get_retrieval_service().search(
            query=payload.query,
            top_k=payload.top_k,
            dataset_name=payload.dataset_name,
            method=payload.method,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SearchResponse(
        query=payload.query,
        method=payload.method,
        dataset_name=payload.dataset_name,
        documents=documents,
        note="Results generated from disk-backed retrieval indexes with lazy loading.",
    )


@router.post("/search/batch", response_model=BatchSearchResponse)
def search_batch(payload: BatchSearchRequest) -> BatchSearchResponse:
    try:
        batch_results = get_retrieval_service().search_batch(
            queries=payload.queries,
            top_k=payload.top_k,
            dataset_name=payload.dataset_name,
            method=payload.method,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results = [
        SearchResponse(
            query=query,
            method=payload.method,
            dataset_name=payload.dataset_name,
            documents=documents,
            note="Batch retrieval result from disk-backed retrieval indexes.",
        )
        for query, documents in zip(payload.queries, batch_results, strict=False)
    ]
    return BatchSearchResponse(
        method=payload.method,
        dataset_name=payload.dataset_name,
        results=results,
        note="Batch retrieval completed successfully.",
    )

