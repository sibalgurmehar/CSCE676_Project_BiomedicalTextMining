from fastapi import APIRouter, HTTPException

from app.schemas import AIPolishRequest, AIPolishResponse
from app.services import get_ai_polish_service

router = APIRouter()


@router.post("/ai-polish", response_model=AIPolishResponse)
def ai_polish(payload: AIPolishRequest) -> AIPolishResponse:
    try:
        response = get_ai_polish_service().polish(
            query=payload.query,
            method=payload.method,
            dataset_name=payload.dataset_name,
            top_k=payload.top_k,
            refine_clusters=payload.refine_clusters,
            include_matches=payload.include_matches,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AIPolishResponse(**response)
