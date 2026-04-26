import sys
from pathlib import Path


TEST_QUERIES = [
    "heart attack",
    "myocardial infarction",
    "diabetes treatment",
    "drug adverse effects",
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    backend_root = project_root / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    from app.retrieval.service import get_retrieval_service

    service = get_retrieval_service()
    datasets = ["unified", "pubmed_subset"]
    methods = ["tfidf", "bm25", "dense", "hybrid"]

    for dataset_name in datasets:
        print(f"\nDATASET: {dataset_name}")
        for method in methods:
            print(f"\nMETHOD: {method}")
            for query in TEST_QUERIES:
                try:
                    results = service.search(
                        query=query,
                        top_k=3,
                        dataset_name=dataset_name,
                        method=method,
                    )
                    pmids = [result.pmid for result in results]
                    print(f"  {query!r} -> {pmids}")
                except Exception as exc:
                    print(f"  {query!r} -> ERROR: {exc}")


if __name__ == "__main__":
    main()
