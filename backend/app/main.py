from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import ai_router, analysis_router, dataset_router, health_router, search_router

app = FastAPI(
    title="BioSeek API",
    description="Backend API for biomedical retrieval method comparison.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(ai_router, prefix="/api", tags=["ai"])
app.include_router(search_router, prefix="/api", tags=["search"])
app.include_router(analysis_router, prefix="/api", tags=["analysis"])
app.include_router(dataset_router, prefix="/api", tags=["dataset"])


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "BioSeek API is running."}
