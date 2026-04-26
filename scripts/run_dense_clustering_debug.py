import argparse
import gc
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


LOGGER = logging.getLogger("bioseek.dense_clustering_debug")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense-only clustering debug runner on a small document sample."
    )
    parser.add_argument("--dataset-name", default="unified", choices=["unified", "pubmed_subset"])
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/clusters/dense_debug"),
    )
    return parser.parse_args()


def get_docs_path(project_root: Path, dataset_name: str) -> Path:
    return project_root / "data" / "processed" / dataset_name / "docs.parquet"


def representative_keywords(texts: list[str], top_terms: int = 6) -> list[str]:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    matrix = vectorizer.fit_transform(texts)
    centroid_values = np.asarray(matrix.mean(axis=0)).ravel()
    feature_names = vectorizer.get_feature_names_out()
    top_idx = np.argsort(centroid_values)[::-1][:top_terms]
    return [str(feature_names[idx]) for idx in top_idx if centroid_values[idx] > 0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    docs_path = get_docs_path(project_root, args.dataset_name)
    docs_df = pd.read_parquet(docs_path).head(args.sample_size).copy()
    docs_df["pmid"] = docs_df["pmid"].astype("string")
    docs_df["retrieval_text"] = docs_df["retrieval_text"].astype("string")

    output_dir = (project_root / args.output_dir / args.dataset_name / f"sample_{args.sample_size}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Dense debug start")
    LOGGER.info("Document count: %s", len(docs_df))
    LOGGER.info("Model loading start: %s", args.model_name)
    try:
        model = SentenceTransformer(args.model_name, device="cpu", local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Dense debug model load failed in local-files-only mode. "
            "Make sure the sentence-transformer model is already cached locally."
        ) from exc
    LOGGER.info("Model loading end")

    texts = docs_df["retrieval_text"].tolist()
    LOGGER.info("Embedding generation start")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    LOGGER.info("Embedding generation end")
    LOGGER.info("Embedding shape: %s", embeddings.shape)

    embeddings_path = output_dir / "dense_embeddings.npy"
    np.save(embeddings_path, embeddings)
    docs_out_path = output_dir / "dense_input_docs.parquet"
    docs_df.to_parquet(docs_out_path, index=False)
    LOGGER.info("Wrote embeddings: %s", embeddings_path)
    LOGGER.info("Wrote docs: %s", docs_out_path)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(docs_df):
        raise ValueError(f"Invalid embedding shape: {embeddings.shape}")

    actual_clusters = max(2, min(args.n_clusters, len(docs_df)))
    LOGGER.info("Clustering start with n_clusters=%s", actual_clusters)
    kmeans = MiniBatchKMeans(
        n_clusters=actual_clusters,
        random_state=42,
        batch_size=min(128, len(docs_df)),
        n_init="auto",
    )
    labels = kmeans.fit_predict(embeddings)
    LOGGER.info("Clustering end")

    assignments_path = output_dir / "raw_cluster_assignments.parquet"
    assignments_df = pd.DataFrame({"pmid": docs_df["pmid"], "cluster_id": labels.astype(int)})
    assignments_df.to_parquet(assignments_path, index=False)
    LOGGER.info("Wrote cluster assignments: %s", assignments_path)

    silhouette = None
    if len(set(labels.tolist())) > 1 and len(docs_df) > actual_clusters:
        silhouette = float(silhouette_score(embeddings, labels))

    cluster_summaries = []
    size_dist: dict[str, int] = {}
    for cluster_id in range(actual_clusters):
        cluster_df = docs_df.iloc[np.where(labels == cluster_id)[0]].copy()
        size_dist[str(cluster_id)] = int(len(cluster_df))
        texts = cluster_df["retrieval_text"].tolist()
        cluster_summaries.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": int(len(cluster_df)),
                "representative_keywords": representative_keywords(texts) if texts else [],
                "representative_docs": cluster_df.head(3)[["pmid", "title", "journal", "year"]].to_dict(orient="records"),
            }
        )

    summary = {
        "dataset_name": args.dataset_name,
        "sample_size": int(len(docs_df)),
        "embedding_shape": list(embeddings.shape),
        "cluster_count": actual_clusters,
        "silhouette_score": silhouette,
        "cluster_size_distribution": size_dist,
        "cluster_summaries": cluster_summaries,
        "status": "success",
    }
    summary_path = output_dir / "dense_clustering_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote summary: %s", summary_path)
    LOGGER.info("Dense debug complete")

    del embeddings
    gc.collect()

    print(f"Doc count: {len(docs_df)}")
    print(f"Embedding shape: {summary['embedding_shape']}")
    print(f"Cluster count: {actual_clusters}")
    print(f"Outputs written: {summary_path.exists() and embeddings_path.exists() and docs_out_path.exists() and assignments_path.exists()}")


if __name__ == "__main__":
    main()
