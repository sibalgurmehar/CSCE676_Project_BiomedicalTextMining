import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def precision_at_k(retrieved_pmids: list[str], relevant_pmids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_pmids[:k]
    if not top:
        return 0.0
    hits = sum(1 for pmid in top if pmid in relevant_pmids)
    return hits / k


def recall_at_k(retrieved_pmids: list[str], relevant_pmids: set[str], k: int) -> float:
    if not relevant_pmids:
        return 0.0
    hits = sum(1 for pmid in retrieved_pmids[:k] if pmid in relevant_pmids)
    return hits / len(relevant_pmids)


def reciprocal_rank(retrieved_pmids: list[str], relevant_pmids: set[str]) -> float:
    for rank, pmid in enumerate(retrieved_pmids, start=1):
        if pmid in relevant_pmids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_pmids: list[str], relevant_pmids: set[str], k: int) -> float:
    if not relevant_pmids:
        return 0.0
    dcg = 0.0
    for rank, pmid in enumerate(retrieved_pmids[:k], start=1):
        rel = 1.0 if pmid in relevant_pmids else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)
    ideal_hits = min(len(relevant_pmids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (project_root / "eval/configs/dense_adaptation_config.json").read_text(encoding="utf-8")
    )

    backend_root = project_root / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    from app.retrieval.dense import DenseRetriever

    questions_df = pd.read_parquet((project_root / config["questions_path"]).resolve())
    qrels_df = pd.read_parquet((project_root / config["qrels_path"]).resolve())
    qrels_map: dict[str, set[str]] = defaultdict(set)
    for row in qrels_df.to_dict(orient="records"):
        qrels_map[str(row["query_id"])].add(str(row["pmid"]))

    base_retriever = DenseRetriever(
        dataset_name=config["evaluation_dataset_name"],
        model_name=config["base_model_name"],
        index_name="dense",
    )
    adapted_model_dir = str((project_root / config["output_model_dir"]).resolve())
    adapted_retriever = DenseRetriever(
        dataset_name=config["evaluation_dataset_name"],
        model_name=adapted_model_dir,
        index_name="adapted_dense",
    )

    methods = {
        "base_dense": base_retriever,
        "adapted_dense": adapted_retriever,
    }
    rows = []
    top_k = int(config["top_k"])
    for method_name, retriever in methods.items():
        for query in questions_df.to_dict(orient="records"):
            query_id = str(query["query_id"])
            query_text = str(query["query_text"])
            relevant_pmids = qrels_map.get(query_id, set())
            results = retriever.search(query=query_text, top_k=top_k)
            retrieved_pmids = [result.pmid for result in results]
            rows.append(
                {
                    "method": method_name,
                    "query_id": query_id,
                    "precision_at_10": precision_at_k(retrieved_pmids, relevant_pmids, 10),
                    "recall_at_10": recall_at_k(retrieved_pmids, relevant_pmids, 10),
                    "mrr": reciprocal_rank(retrieved_pmids, relevant_pmids),
                    "ndcg_at_10": ndcg_at_k(retrieved_pmids, relevant_pmids, 10),
                }
            )

    metrics_df = pd.DataFrame(rows)
    summary_df = (
        metrics_df.groupby("method")[["precision_at_10", "recall_at_10", "mrr", "ndcg_at_10"]]
        .mean()
        .reset_index()
    )

    base_row = summary_df[summary_df["method"] == "base_dense"].iloc[0].to_dict()
    adapted_row = summary_df[summary_df["method"] == "adapted_dense"].iloc[0].to_dict()
    delta_ndcg = float(adapted_row["ndcg_at_10"] - base_row["ndcg_at_10"])

    if delta_ndcg > 0.005:
        takeaway = (
            f"The adapted dense retriever improved nDCG@10 by {delta_ndcg:.4f} over the base dense model "
            "on the unified dataset."
        )
    elif delta_ndcg < -0.005:
        takeaway = (
            f"The adapted dense retriever underperformed the base dense model by {abs(delta_ndcg):.4f} nDCG@10. "
            "This lightweight adaptation did not help and should be reported honestly as a negative result."
        )
    else:
        takeaway = (
            f"The adapted dense retriever changed nDCG@10 by only {delta_ndcg:.4f}. "
            "This suggests the lightweight adaptation had little practical effect."
        )

    report = {
        "config": config,
        "summary": summary_df.to_dict(orient="records"),
        "takeaway": takeaway,
        "honesty_note": "If the adapted model does not improve the dense baseline, keep the base model in the main project and present adaptation as an optional negative or neutral experiment.",
    }

    report_path = (project_root / config["comparison_report_path"]).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    metrics_df.to_parquet(report_path.with_name("comparison_per_query.parquet"), index=False)

    print("Stage 7 comparison complete")
    print(f"Report: {report_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

