import argparse
import json
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster top retrieved PubMed documents for a query."
    )
    parser.add_argument("--query", required=True, help="Search query to retrieve and cluster.")
    parser.add_argument("--method", default="bm25", help="Retrieval method: tfidf, bm25, dense, hybrid.")
    parser.add_argument("--dataset-name", default="unified", help="Dataset name: unified or pubmed_subset.")
    parser.add_argument("--top-k", type=int, default=30, help="Number of retrieved docs to cluster.")
    parser.add_argument("--num-clusters", type=int, default=5, help="Target number of clusters.")
    parser.add_argument(
        "--vector-space",
        default="tfidf",
        choices=["tfidf", "dense"],
        help="Vector space used for clustering.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to eval/results/clustering/<slug>.json",
    )
    return parser.parse_args()


def result_to_dict(result):
    return {
        "pmid": result.pmid,
        "score": result.score,
        "retrieval_text": result.retrieval_text,
        "metadata": result.metadata,
        "method": result.method,
        "dataset_name": result.dataset_name,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    project_root = Path(__file__).resolve().parents[1]
    backend_root = project_root / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    from app.clustering.service import get_clustering_service

    args = parse_args()
    output = get_clustering_service().cluster_search_results(
        query=args.query,
        method=args.method,
        dataset_name=args.dataset_name,
        top_k=args.top_k,
        num_clusters=args.num_clusters,
        vector_space=args.vector_space,
    )

    payload = {
        "query": output.query,
        "dataset_name": output.dataset_name,
        "method": output.method,
        "vector_space": output.vector_space,
        "num_clusters": output.num_clusters,
        "status": output.status,
        "error": output.error,
        "artifact_paths": output.artifact_paths,
        "silhouette_score": output.silhouette_score,
        "cluster_size_distribution": output.cluster_size_distribution,
        "representative_term_summaries": output.representative_term_summaries,
        "cluster_summaries": [
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(output.cluster_size_distribution[cluster_id]),
                "representative_keywords": output.representative_term_summaries[cluster_id],
                "representative_docs": [
                    result_to_dict(result) for result in results[:3]
                ],
            }
            for cluster_id, results in sorted(
                output.grouped_results.items(),
                key=lambda item: int(item[0]),
            )
        ],
        "grouped_results": {
            cluster_id: [result_to_dict(result) for result in results]
            for cluster_id, results in output.grouped_results.items()
        },
        "diversified_results": [result_to_dict(result) for result in output.diversified_results],
    }

    if args.output_path is None:
        slug = (
            args.query.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("?", "")
            .replace(",", "")
        )
        output_path = project_root / "eval" / "results" / "clustering" / f"{slug}_{args.method}_{args.vector_space}.json"
    else:
        output_path = (project_root / args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Stage 8 clustering complete")
    print(f"Output: {output_path}")
    print(f"Silhouette score: {output.silhouette_score}")
    print(f"Cluster size distribution: {output.cluster_size_distribution}")


if __name__ == "__main__":
    main()
