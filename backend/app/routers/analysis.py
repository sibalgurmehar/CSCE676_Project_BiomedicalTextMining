from fastapi import APIRouter, HTTPException

from app.schemas import (
    ArtifactResponse,
    QueryAnalysisRequest,
    QueryAnalysisResponse,
    QueryMetricsRequest,
    QueryMetricsResponse,
    SummaryResponse,
)
from app.services import get_analysis_service, get_query_service

router = APIRouter()


@router.post("/query-analysis", response_model=QueryAnalysisResponse)
def query_analysis(payload: QueryAnalysisRequest) -> QueryAnalysisResponse:
    try:
        bucket_rule_based, bucket_final, rationale = get_query_service().classify_query(payload.query)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return QueryAnalysisResponse(
        query=payload.query,
        bucket_rule_based=bucket_rule_based,
        bucket_final=bucket_final,
        rationale=rationale,
    )


@router.get("/metrics/summary", response_model=SummaryResponse)
def metrics_summary() -> SummaryResponse:
    try:
        payload = get_analysis_service().metrics_summary()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SummaryResponse(**payload)


@router.get("/diversity/summary", response_model=SummaryResponse)
def diversity_summary() -> SummaryResponse:
    try:
        payload = get_analysis_service().diversity_summary()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SummaryResponse(**payload)


@router.get("/query-types/summary", response_model=SummaryResponse)
def query_types_summary() -> SummaryResponse:
    try:
        payload = get_analysis_service().query_type_summary()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SummaryResponse(**payload)


@router.get("/metrics/examples", response_model=ArtifactResponse)
def metrics_examples() -> ArtifactResponse:
    try:
        payload = get_analysis_service().best_worst_queries()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ArtifactResponse(**payload)


@router.post("/metrics/query", response_model=QueryMetricsResponse)
def metrics_query(payload: QueryMetricsRequest) -> QueryMetricsResponse:
    try:
        response = get_analysis_service().query_metrics(payload.query)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return QueryMetricsResponse(**response)
