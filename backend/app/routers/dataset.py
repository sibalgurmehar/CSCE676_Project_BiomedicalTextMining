from fastapi import APIRouter, HTTPException

from app.schemas import DatasetInfoResponse
from app.services import get_dataset_service

router = APIRouter()


@router.get("/dataset-info", response_model=DatasetInfoResponse)
def dataset_info() -> DatasetInfoResponse:
    try:
        payload = get_dataset_service().dataset_info()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DatasetInfoResponse(**payload)

