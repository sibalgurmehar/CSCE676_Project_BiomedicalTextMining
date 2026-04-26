import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BioSeek retrieval indexes for the curated datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["unified", "pubmed_subset"],
        help="Datasets to index.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["tfidf", "bm25"],
        help="Retrieval methods to build.",
    )
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    backend_root = project_root / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    from app.retrieval.service import get_retrieval_service

    args = parse_args()
    service = get_retrieval_service()
    for dataset_name in args.datasets:
        print(f"Building indexes for dataset={dataset_name}")
        service.build_indexes(dataset_name=dataset_name, methods=args.methods)
        print(f"Completed dataset={dataset_name} methods={args.methods}")


if __name__ == "__main__":
    main()
