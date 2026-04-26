from fastapi import APIRouter, HTTPException

from app.schemas import (
    ClusterRequest,
    ClusterResponse,
    CompareRequest,
    CompareResponse,
    MethodResults,
    SearchRequest,
    SearchResponse,
)
from app.services import get_search_service

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    try:
        documents = get_search_service().search(
            query=payload.query,
            method=payload.method,
            dataset_name=payload.dataset_name,
            top_k=payload.top_k,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SearchResponse(
        query=payload.query,
        method=payload.method,
        dataset_name=payload.dataset_name,
        documents=documents,
        note="Search uses the curated PubMed subset by default for exploration and local experimentation.",
    )


@router.post("/compare", response_model=CompareResponse)
def compare(payload: CompareRequest) -> CompareResponse:
    try:
        results = get_search_service().compare(
            query=payload.query,
            methods=payload.methods,
            dataset_name=payload.dataset_name,
            top_k=payload.top_k,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CompareResponse(
        query=payload.query,
        dataset_name=payload.dataset_name,
        results=[MethodResults(**result) for result in results],
        note="Comparison uses the unified dataset by default for evaluation-driven retrieval comparison.",
    )


@router.post("/clusters", response_model=ClusterResponse)
def clusters(payload: ClusterRequest) -> ClusterResponse:
    try:
        return get_search_service().clusters(
            query=payload.query,
            method=payload.method,
            dataset_name=payload.dataset_name,
            top_k=payload.top_k,
            num_clusters=payload.num_clusters,
            vector_space=payload.vector_space,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

