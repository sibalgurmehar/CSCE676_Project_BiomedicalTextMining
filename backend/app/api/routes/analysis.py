from fastapi import APIRouter

router = APIRouter()


@router.get("/analysis/diversity")
def diversity_analysis() -> dict:
    return {
        "overlap_rate": {
            "tfidf_bm25": 0.62,
            "tfidf_dense": 0.28,
            "bm25_hybrid": 0.57,
        },
        "cluster_coverage": {
            "tfidf": 3,
            "bm25": 3,
            "dense": 4,
            "hybrid": 4,
        },
        "note": "Placeholder analysis output for frontend integration and course-demo wiring.",
    }

