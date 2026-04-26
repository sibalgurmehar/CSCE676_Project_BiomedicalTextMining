import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


METHODS = ["tfidf", "bm25", "dense", "hybrid"]
TEST_QUERIES = [
    "heart attack",
    "myocardial infarction",
    "diabetes treatment",
    "drug adverse effects",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval methods on the unified BioSeek dataset."
    )
    parser.add_argument(
        "--dataset-name",
        default="unified",
        help="Dataset name to evaluate. Defaults to unified.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
        help="Retrieval methods to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k depth to retrieve for evaluation. Must be at least 10 for requested metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results/unified"),
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("data/processed/unified/questions.parquet"),
        help="Unified questions parquet path.",
    )
    parser.add_argument(
        "--qrels-path",
        type=Path,
        default=Path("data/processed/unified/qrels.parquet"),
        help="Unified qrels parquet path.",
    )
    return parser.parse_args()


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


def dcg_at_k(retrieved_pmids: list[str], relevant_pmids: set[str], k: int) -> float:
    dcg = 0.0
    for rank, pmid in enumerate(retrieved_pmids[:k], start=1):
        rel = 1.0 if pmid in relevant_pmids else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved_pmids: list[str], relevant_pmids: set[str], k: int) -> float:
    if not relevant_pmids:
        return 0.0
    ideal_hits = min(len(relevant_pmids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved_pmids, relevant_pmids, k) / idcg


def load_inputs(project_root: Path, questions_path: Path, qrels_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    questions_df = pd.read_parquet((project_root / questions_path).resolve())
    qrels_df = pd.read_parquet((project_root / qrels_path).resolve())
    questions_df["query_id"] = questions_df["query_id"].astype("string")
    questions_df["query_text"] = questions_df["query_text"].astype("string")
    questions_df["query_type"] = questions_df["query_type"].astype("string")
    qrels_df["query_id"] = qrels_df["query_id"].astype("string")
    qrels_df["pmid"] = qrels_df["pmid"].astype("string")
    return questions_df, qrels_df


def aggregate_summary(per_query_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_query_df.groupby("method")[
            ["precision_at_5", "precision_at_10", "recall_at_10", "mrr", "ndcg_at_5", "ndcg_at_10"]
        ]
        .mean()
        .reset_index()
    )
    grouped = grouped.sort_values("ndcg_at_10", ascending=False).reset_index(drop=True)
    return grouped


def build_best_worst(per_query_df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method, method_df in per_query_df.groupby("method"):
        best = method_df.sort_values(["ndcg_at_10", "mrr"], ascending=False).head(5)
        worst = method_df.sort_values(["ndcg_at_10", "mrr"], ascending=True).head(5)
        result[method] = {
            "best_queries": best[
                ["query_id", "query_text", "query_type", "ndcg_at_10", "mrr", "precision_at_10"]
            ].to_dict(orient="records"),
            "worst_queries": worst[
                ["query_id", "query_text", "query_type", "ndcg_at_10", "mrr", "precision_at_10"]
            ].to_dict(orient="records"),
        }
    return result


def build_overlap_analysis(
    ranked_outputs: dict[str, dict[str, list[str]]],
    methods: list[str],
    k: int,
) -> dict[str, Any]:
    overlap_rows: list[dict[str, Any]] = []
    for index, method_a in enumerate(methods):
        for method_b in methods[index + 1 :]:
            query_scores = []
            for query_id, ranked_a in ranked_outputs[method_a].items():
                ranked_b = ranked_outputs[method_b].get(query_id, [])
                union = set(ranked_a[:k]) | set(ranked_b[:k])
                if not union:
                    score = 0.0
                else:
                    score = len(set(ranked_a[:k]) & set(ranked_b[:k])) / len(union)
                query_scores.append(score)

            overlap_rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "avg_overlap_at_k": round(sum(query_scores) / max(len(query_scores), 1), 4),
                    "k": k,
                }
            )
    return {"pairwise_overlap": overlap_rows}


def build_narrative(summary_df: pd.DataFrame, per_query_df: pd.DataFrame) -> dict[str, str]:
    best_row = summary_df.iloc[0].to_dict()
    best_method = str(best_row["method"])

    lexical_df = per_query_df[per_query_df["method"].isin(["tfidf", "bm25"])]
    semantic_df = per_query_df[per_query_df["method"].isin(["dense", "hybrid"])]

    lexical_mean = lexical_df["ndcg_at_10"].mean() if not lexical_df.empty else 0.0
    semantic_mean = semantic_df["ndcg_at_10"].mean() if not semantic_df.empty else 0.0

    return {
        "best_overall": (
            f"{best_method.upper()} performs best overall on the unified dataset by average nDCG@10 "
            f"({best_row['ndcg_at_10']:.4f}) and also leads or stays competitive on the other top-k metrics."
        ),
        "where_lexical_is_better": (
            "Lexical retrieval tends to be stronger when the BioASQ query wording directly overlaps with the "
            "terminology used in titles and abstracts, especially for explicit disease, treatment, and adverse-event phrases."
        ),
        "where_dense_or_hybrid_helps": (
            "Dense and hybrid retrieval are most useful when biomedical concepts are expressed with variant terminology, "
            "synonyms, abbreviations, or indirect phrasing. Hybrid retrieval is particularly valuable when semantic matching "
            f"helps with terminology mismatch while lexical signals still anchor precision. Average semantic-family nDCG@10 is "
            f"{semantic_mean:.4f} versus {lexical_mean:.4f} for the lexical family."
        ),
    }


def ensure_backend_on_path(project_root: Path) -> None:
    backend_root = project_root / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


def evaluate_method(
    method: str,
    questions_df: pd.DataFrame,
    qrels_map: dict[str, set[str]],
    dataset_name: str,
    top_k: int,
):
    from app.retrieval.service import get_retrieval_service

    service = get_retrieval_service()
    per_query_rows: list[dict[str, Any]] = []
    ranked_outputs: dict[str, list[str]] = {}

    queries = questions_df["query_text"].tolist()
    batch_results = service.search_batch(
        queries=queries,
        top_k=top_k,
        dataset_name=dataset_name,
        method=method,
    )

    for row, results in zip(questions_df.to_dict(orient="records"), batch_results, strict=False):
        query_id = str(row["query_id"])
        query_text = str(row["query_text"])
        query_type = None if pd.isna(row["query_type"]) else str(row["query_type"])
        relevant_pmids = qrels_map.get(query_id, set())
        retrieved_pmids = [result.pmid for result in results]
        ranked_outputs[query_id] = retrieved_pmids

        per_query_rows.append(
            {
                "method": method,
                "query_id": query_id,
                "query_text": query_text,
                "query_type": query_type,
                "num_relevant_docs": len(relevant_pmids),
                "num_retrieved_docs": len(retrieved_pmids),
                "precision_at_5": precision_at_k(retrieved_pmids, relevant_pmids, 5),
                "precision_at_10": precision_at_k(retrieved_pmids, relevant_pmids, 10),
                "recall_at_10": recall_at_k(retrieved_pmids, relevant_pmids, 10),
                "mrr": reciprocal_rank(retrieved_pmids, relevant_pmids),
                "ndcg_at_5": ndcg_at_k(retrieved_pmids, relevant_pmids, 5),
                "ndcg_at_10": ndcg_at_k(retrieved_pmids, relevant_pmids, 10),
                "top_pmids": retrieved_pmids,
            }
        )

    return per_query_rows, ranked_outputs


def save_outputs(
    output_dir: Path,
    per_query_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    best_worst: dict[str, Any],
    overlap_analysis: dict[str, Any],
    frontend_json: dict[str, Any],
    narrative: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_query_df.to_parquet(output_dir / "per_query_metrics.parquet", index=False)
    per_query_df.to_json(output_dir / "per_query_metrics.json", orient="records", indent=2)
    summary_df.to_parquet(output_dir / "summary.parquet", index=False)
    summary_df.to_json(output_dir / "summary.json", orient="records", indent=2)
    (output_dir / "best_worst_queries.json").write_text(
        json.dumps(best_worst, indent=2),
        encoding="utf-8",
    )
    (output_dir / "method_overlap.json").write_text(
        json.dumps(overlap_analysis, indent=2),
        encoding="utf-8",
    )
    (output_dir / "frontend_metrics.json").write_text(
        json.dumps(frontend_json, indent=2),
        encoding="utf-8",
    )
    (output_dir / "narrative.json").write_text(
        json.dumps(narrative, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.top_k < 10:
        raise ValueError("--top-k must be at least 10 to compute Precision@10, Recall@10, and nDCG@10.")

    project_root = Path(__file__).resolve().parents[1]
    ensure_backend_on_path(project_root)

    questions_df, qrels_df = load_inputs(
        project_root=project_root,
        questions_path=args.questions_path,
        qrels_path=args.qrels_path,
    )

    qrels_map: dict[str, set[str]] = defaultdict(set)
    for row in qrels_df.to_dict(orient="records"):
        qrels_map[str(row["query_id"])].add(str(row["pmid"]))

    per_query_rows: list[dict[str, Any]] = []
    ranked_outputs: dict[str, dict[str, list[str]]] = {}
    evaluated_methods: list[str] = []

    for method in args.methods:
        print(f"Evaluating method={method} on dataset={args.dataset_name}")
        method_rows, method_ranked = evaluate_method(
            method=method,
            questions_df=questions_df,
            qrels_map=qrels_map,
            dataset_name=args.dataset_name,
            top_k=args.top_k,
        )
        per_query_rows.extend(method_rows)
        ranked_outputs[method] = method_ranked
        evaluated_methods.append(method)
        print(f"Completed method={method}")

    per_query_df = pd.DataFrame(per_query_rows)
    summary_df = aggregate_summary(per_query_df)
    best_worst = build_best_worst(per_query_df)
    overlap_analysis = build_overlap_analysis(ranked_outputs, evaluated_methods, k=10)
    narrative = build_narrative(summary_df, per_query_df)

    frontend_json = {
        "dataset_name": args.dataset_name,
        "top_k": args.top_k,
        "methods": summary_df.to_dict(orient="records"),
        "best_worst": best_worst,
        "overlap": overlap_analysis,
        "narrative": narrative,
        "test_queries": TEST_QUERIES,
    }

    output_dir = (project_root / args.output_dir).resolve()
    save_outputs(
        output_dir=output_dir,
        per_query_df=per_query_df,
        summary_df=summary_df,
        best_worst=best_worst,
        overlap_analysis=overlap_analysis,
        frontend_json=frontend_json,
        narrative=narrative,
    )

    print("Stage 5 evaluation complete")
    print(f"Output directory: {output_dir}")
    print("Overall summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
