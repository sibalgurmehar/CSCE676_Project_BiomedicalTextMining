from app.routers.ai import router as ai_router
from app.routers.analysis import router as analysis_router
from app.routers.dataset import router as dataset_router
from app.routers.health import router as health_router
from app.routers.search import router as search_router

__all__ = ["ai_router", "analysis_router", "dataset_router", "health_router", "search_router"]
