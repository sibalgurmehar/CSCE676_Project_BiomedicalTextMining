from fastapi import APIRouter

from app.core.settings import get_project_settings

router = APIRouter()


@router.get("/config")
def config() -> dict:
    settings = get_project_settings()
    return settings.model_dump()


@router.get("/methods")
def methods() -> list[dict]:
    settings = get_project_settings()
    return [method.model_dump() for method in settings.retrieval_methods]

