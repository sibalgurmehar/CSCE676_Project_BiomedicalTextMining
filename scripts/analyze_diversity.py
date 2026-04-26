import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze retrieval diversity over top-k results using global corpus clusters."
    )
    parser.add_argument("--dataset-name", default="unified")
    parser.add_argument("--vector-space", choices=["tfidf", "dense"], default="dense")
    parser.add_argument("--num-clusters", type=int, default=40)
    parser.add_argument(
        "--per-query-metrics-path",
        type=Path,
        default=Path("eval/results/unified/per_query_metrics.parquet"),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("eval/results/unified/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results/unified/diversity"),
    )
    return parser.parse_args()


def load_corpus_assets(project_root: Path, dataset_name: str, vector_space: str):
    docs_path = project_root / "data" / "processed" / dataset_name / "docs.parquet"
    docs_df = pd.read_parquet(docs_path)
    docs_df["pmid"] = docs_df["pmid"].astype("string")
    docs_df["retrieval_text"] = docs_df["retrieval_text"].astype("string")
    index_dir = project_root / "data" / "indexes" / dataset_name / vector_space

    if vector_space == "dense":
        vectors = np.load(index_dir / "embeddings.npy")
        shared_vectorizer = None
    else:
        from scipy import sparse

        vectors = sparse.load_npz(index_dir / "matrix.npz").tocsr()
        import pickle

        with (index_dir / "vectorizer.pkl").open("rb") as handle:
            shared_vectorizer = pickle.load(handle)
    return docs_df, vectors, shared_vectorizer


def cluster_corpus(vectors, num_clusters: int):
    actual_clusters = max(2, min(num_clusters, vectors.shape[0]))
    kmeans = MiniBatchKMeans(
        n_clusters=actual_clusters,
        random_state=42,
        batch_size=min(1024, vectors.shape[0]),
        n_init="auto",
    )
    labels = kmeans.fit_predict(vectors)
    return labels, actual_clusters


def representative_keywords_for_cluster(
    cluster_texts: list[str],
    shared_vectorizer=None,
    cluster_vectors=None,
    top_terms: int = 6,
) -> list[str]:
    if shared_vectorizer is not None and cluster_vectors is not None:
        centroid = cluster_vectors.mean(axis=0)
        values = centroid.A1 if hasattr(centroid, "A1") else np.asarray(centroid).ravel()
        feature_names = shared_vectorizer.get_feature_names_out()
        top_idx = np.argsort(values)[::-1][:top_terms]
        return [str(feature_names[idx]) for idx in top_idx if values[idx] > 0]

    local_vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    matrix = local_vectorizer.fit_transform(cluster_texts)
    values = np.asarray(matrix.mean(axis=0)).ravel()
    feature_names = local_vectorizer.get_feature_names_out()
    top_idx = np.argsort(values)[::-1][:top_terms]
    return [str(feature_names[idx]) for idx in top_idx if values[idx] > 0]


def build_global_cluster_artifacts(
    docs_df: pd.DataFrame,
    vectors,
    shared_vectorizer,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assignments_df = pd.DataFrame(
        {
            "pmid": docs_df["pmid"].astype("string"),
            "cluster_id": labels.astype(int),
        }
    )

    cluster_summary: dict[str, Any] = {}
    for cluster_id in sorted(assignments_df["cluster_id"].unique().tolist()):
        member_positions = assignments_df.index[assignments_df["cluster_id"] == cluster_id].tolist()
        cluster_texts = docs_df.iloc[member_positions]["retrieval_text"].astype(str).tolist()
        cluster_vectors = vectors[member_positions] if vectors is not None else None
        keywords = representative_keywords_for_cluster(
            cluster_texts=cluster_texts,
            shared_vectorizer=shared_vectorizer,
            cluster_vectors=cluster_vectors,
        )
        representative_docs = docs_df.iloc[member_positions].head(3)[
            ["pmid", "title", "journal", "year"]
        ].to_dict(orient="records")
        cluster_summary[str(cluster_id)] = {
            "cluster_id": int(cluster_id),
            "cluster_size": int(len(member_positions)),
            "representative_keywords": keywords,
            "representative_docs": representative_docs,
        }
    return assignments_df, cluster_summary


def normalized_entropy(cluster_ids: list[int]) -> float:
    if not cluster_ids:
        return 0.0
    counts = pd.Series(cluster_ids).value_counts(normalize=True)
    entropy = -sum(float(p) * math.log(float(p), 2) for p in counts.tolist() if p > 0)
    max_entropy = math.log(len(counts), 2) if len(counts) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def per_query_diversity(
    per_query_df: pd.DataFrame,
    pmid_to_cluster: dict[str, int],
    cluster_summary: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in per_query_df.to_dict(orient="records"):
        top_pmids = [str(pmid) for pmid in row["top_pmids"] if str(pmid) in pmid_to_cluster]
        cluster_ids = [pmid_to_cluster[pmid] for pmid in top_pmids]
        distinct_cluster_ids = sorted(set(cluster_ids))
        theme_keywords = sorted(
            {
                keyword
                for cluster_id in distinct_cluster_ids
                for keyword in cluster_summary[str(cluster_id)]["representative_keywords"]
            }
        )
        rows.append(
            {
                "query_id": row["query_id"],
                "query_text": row["query_text"],
                "query_type": row["query_type"],
                "method": row["method"],
                "clusters_represented": len(distinct_cluster_ids),
                "cluster_entropy": normalized_entropy(cluster_ids),
                "theme_coverage": len(theme_keywords),
                "represented_cluster_ids": distinct_cluster_ids,
                "theme_keywords": theme_keywords,
                "num_retrieved_docs": len(top_pmids),
            }
        )
    return pd.DataFrame(rows)


def build_method_comparisons(diversity_df: pd.DataFrame, relevance_summary: list[dict[str, Any]]) -> pd.DataFrame:
    diversity_summary = (
        diversity_df.groupby("method")[["clusters_represented", "cluster_entropy", "theme_coverage"]]
        .mean()
        .reset_index()
    )
    relevance_df = pd.DataFrame(relevance_summary)
    return diversity_summary.merge(relevance_df, on="method", how="left")


def build_narrative(method_df: pd.DataFrame) -> dict[str, str]:
    precise = method_df.sort_values(["precision_at_10", "cluster_entropy"], ascending=[False, True]).iloc[0]
    broad = method_df.sort_values(["theme_coverage", "cluster_entropy"], ascending=[False, False]).iloc[0]
    hybrid_row = method_df[method_df["method"] == "hybrid"].iloc[0] if "hybrid" in method_df["method"].tolist() else None
    best_diverse = method_df.sort_values(["ndcg_at_10", "cluster_entropy"], ascending=[False, False]).iloc[0]

    hybrid_line = (
        f"Hybrid shows precision@10={hybrid_row['precision_at_10']:.4f}, "
        f"nDCG@10={hybrid_row['ndcg_at_10']:.4f}, cluster_entropy={hybrid_row['cluster_entropy']:.4f}, "
        f"and theme_coverage={hybrid_row['theme_coverage']:.2f}, suggesting a balance between relevance and breadth."
        if hybrid_row is not None
        else "Hybrid was not included in this run."
    )

    return {
        "narrow_but_precise": (
            f"{precise['method'].upper()} is the narrowest strong performer in this run, combining higher precision "
            f"(P@10={precise['precision_at_10']:.4f}) with lower spread than broader methods."
        ),
        "broader_theme_coverage": (
            f"{broad['method'].upper()} covers the broadest themes in this run with average theme coverage "
            f"{broad['theme_coverage']:.2f} and cluster entropy {broad['cluster_entropy']:.4f}."
        ),
        "hybrid_balance": hybrid_line,
        "best_relevance_diversity_tradeoff": (
            f"{best_diverse['method'].upper()} offers the strongest combined relevance/diversity profile in this run "
            f"with nDCG@10={best_diverse['ndcg_at_10']:.4f} and cluster entropy={best_diverse['cluster_entropy']:.4f}."
        ),
    }


def ensure_notebook(output_path: Path, relative_csv_path: str) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# BioSeek Diversity Analysis\n", "Visualize diversity metrics by retrieval method.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    f"df = pd.read_csv('{relative_csv_path}')\n",
                    "df\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plot_df = df.set_index('method')[['clusters_represented', 'cluster_entropy', 'theme_coverage']]\n",
                    "plot_df.plot(kind='bar', figsize=(10, 5), title='Method-Level Diversity Metrics')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    docs_df, vectors, shared_vectorizer = load_corpus_assets(
        project_root=project_root,
        dataset_name=args.dataset_name,
        vector_space=args.vector_space,
    )
    labels, actual_clusters = cluster_corpus(vectors=vectors, num_clusters=args.num_clusters)
    assignments_df, cluster_summary = build_global_cluster_artifacts(
        docs_df=docs_df,
        vectors=vectors,
        shared_vectorizer=shared_vectorizer,
        labels=labels,
    )

    per_query_df = pd.read_parquet((project_root / args.per_query_metrics_path).resolve())
    relevance_summary = json.loads((project_root / args.summary_path).resolve().read_text(encoding="utf-8"))

    pmid_to_cluster = {
        str(row["pmid"]): int(row["cluster_id"])
        for row in assignments_df.to_dict(orient="records")
    }
    diversity_df = per_query_diversity(
        per_query_df=per_query_df,
        pmid_to_cluster=pmid_to_cluster,
        cluster_summary=cluster_summary,
    )
    method_df = build_method_comparisons(diversity_df=diversity_df, relevance_summary=relevance_summary)
    narrative = build_narrative(method_df)

    output_dir = (project_root / args.output_dir / args.vector_space).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments_df.to_parquet(output_dir / "global_cluster_assignments.parquet", index=False)
    diversity_df.to_parquet(output_dir / "per_query_diversity.parquet", index=False)
    diversity_df.to_json(output_dir / "per_query_diversity.json", orient="records", indent=2)
    method_df.to_csv(output_dir / "method_level_diversity.csv", index=False)

    diversity_metrics = {
        "dataset_name": args.dataset_name,
        "vector_space": args.vector_space,
        "num_global_clusters": actual_clusters,
        "method_level_diversity": method_df.to_dict(orient="records"),
        "narrative": narrative,
        "global_cluster_summary": cluster_summary,
    }
    (output_dir / "diversity_metrics.json").write_text(
        json.dumps(diversity_metrics, indent=2),
        encoding="utf-8",
    )

    notebook_path = project_root / "notebooks" / f"06_diversity_analysis_{args.vector_space}.ipynb"
    ensure_notebook(
        output_path=notebook_path,
        relative_csv_path=f"../eval/results/unified/diversity/{args.vector_space}/method_level_diversity.csv",
    )

    print("Stage 9 diversity analysis complete")
    print(f"Output directory: {output_dir}")
    print(method_df.to_string(index=False))


if __name__ == "__main__":
    main()
