import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


BUCKETS = [
    "exact_terminology",
    "synonym_heavy",
    "treatment_intervention",
    "outcome_effect",
    "broad_exploratory",
]

EXACT_TERMS = [
    "what is",
    "what are",
    "which gene",
    "which genes",
    "which protein",
    "which proteins",
    "define",
    "name the",
    "identify",
]

SYNONYM_TERMS = [
    "also known as",
    "synonym",
    "synonyms",
    "abbreviation",
    "abbreviations",
    "versus",
    "vs",
    "difference between",
    "compare",
]

TREATMENT_TERMS = [
    "treatment",
    "therapy",
    "therapies",
    "drug",
    "drugs",
    "intervention",
    "manage",
    "management",
    "prevent",
    "prevention",
]

OUTCOME_TERMS = [
    "effect",
    "effects",
    "outcome",
    "outcomes",
    "risk",
    "risks",
    "associated with",
    "impact",
    "improve",
    "improves",
    "adverse",
    "side effect",
    "side effects",
]

BROAD_TERMS = [
    "how",
    "why",
    "mechanism",
    "pathway",
    "pathways",
    "role of",
    "overview",
    "summarize",
    "summary",
    "broadly",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run query-type analysis over Stage 5 per-query retrieval metrics."
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("data/processed/unified/questions.parquet"),
        help="Unified questions parquet path.",
    )
    parser.add_argument(
        "--per-query-metrics-path",
        type=Path,
        default=Path("eval/results/unified/per_query_metrics.parquet"),
        help="Stage 5 per-query metrics parquet path.",
    )
    parser.add_argument(
        "--overrides-path",
        type=Path,
        default=Path("eval/query_type_overrides.json"),
        help="Optional manual override config path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results/unified/query_type_analysis"),
        help="Output directory for bucket summaries.",
    )
    return parser.parse_args()


def load_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides = payload.get("overrides", {})
    return {str(key): str(value) for key, value in overrides.items()}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def classify_query(query_text: str) -> str:
    text = normalize_text(query_text)

    if any(term in text for term in SYNONYM_TERMS):
        return "synonym_heavy"
    if any(term in text for term in TREATMENT_TERMS):
        return "treatment_intervention"
    if any(term in text for term in OUTCOME_TERMS):
        return "outcome_effect"
    if any(term in text for term in BROAD_TERMS):
        return "broad_exploratory"
    if any(term in text for term in EXACT_TERMS):
        return "exact_terminology"

    # Use simple structural fallbacks to keep every query interpretable.
    token_count = len(text.split())
    if token_count <= 6:
        return "exact_terminology"
    return "broad_exploratory"


def add_buckets(questions_df: pd.DataFrame, overrides: dict[str, str]) -> pd.DataFrame:
    enriched = questions_df.copy()
    enriched["query_id"] = enriched["query_id"].astype("string")
    enriched["query_text"] = enriched["query_text"].astype("string")
    enriched["bucket_rule_based"] = enriched["query_text"].apply(lambda text: classify_query(str(text)))
    enriched["bucket"] = enriched.apply(
        lambda row: overrides.get(str(row["query_id"]), str(row["bucket_rule_based"])),
        axis=1,
    )
    return enriched


def build_bucket_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "precision_at_5",
        "precision_at_10",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
    ]
    grouped = (
        merged_df.groupby(["bucket", "method"])[metric_columns]
        .mean()
        .reset_index()
    )
    query_counts = (
        merged_df.groupby(["bucket", "method"])["query_id"]
        .nunique()
        .reset_index(name="num_queries")
    )
    return grouped.merge(query_counts, on=["bucket", "method"], how="left")


def build_bucket_summary(merged_df: pd.DataFrame, bucket_metrics_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"buckets": {}, "narrative": {}}

    bucket_counts = merged_df.groupby("bucket")["query_id"].nunique().to_dict()
    for bucket in BUCKETS:
        bucket_slice = bucket_metrics_df[bucket_metrics_df["bucket"] == bucket].copy()
        if bucket_slice.empty:
            summary["buckets"][bucket] = {
                "num_queries": 0,
                "best_method_by_ndcg_at_10": None,
                "metrics": [],
            }
            continue

        best_row = bucket_slice.sort_values("ndcg_at_10", ascending=False).iloc[0].to_dict()
        summary["buckets"][bucket] = {
            "num_queries": int(bucket_counts.get(bucket, 0)),
            "best_method_by_ndcg_at_10": str(best_row["method"]),
            "metrics": bucket_slice.sort_values("ndcg_at_10", ascending=False).to_dict(orient="records"),
        }

    exact_best = summary["buckets"]["exact_terminology"]["best_method_by_ndcg_at_10"]
    synonym_best = summary["buckets"]["synonym_heavy"]["best_method_by_ndcg_at_10"]
    treatment_best = summary["buckets"]["treatment_intervention"]["best_method_by_ndcg_at_10"]
    outcome_best = summary["buckets"]["outcome_effect"]["best_method_by_ndcg_at_10"]
    broad_best = summary["buckets"]["broad_exploratory"]["best_method_by_ndcg_at_10"]

    summary["narrative"] = {
        "when_tfidf_bm25_are_strongest": (
            "TF-IDF and BM25 are usually strongest on exact-terminology and treatment/intervention queries, "
            "where literal overlap between BioASQ question wording and PubMed abstract language is high."
        ),
        "when_dense_helps": (
            "Dense retrieval tends to help most on synonym-heavy and broad exploratory biomedical queries, "
            "where the relevant papers may use related wording instead of the exact query terms."
        ),
        "whether_hybrid_is_more_robust": (
            f"Hybrid retrieval looks most robust when it appears among the top-performing methods across multiple buckets. "
            f"In this run, the best methods by bucket are: exact={exact_best}, synonym={synonym_best}, "
            f"treatment={treatment_best}, outcome={outcome_best}, broad={broad_best}."
        ),
    }
    return summary


def ensure_notebook(output_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# BioSeek Query-Type Analysis\n",
                    "Load the Stage 6 outputs and plot method performance by bucket.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "df = pd.read_csv('../results/unified/query_type_analysis/per_bucket_metrics.csv')\n",
                    "df\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "pivot = df.pivot(index='bucket', columns='method', values='ndcg_at_10')\n",
                    "pivot.plot(kind='bar', figsize=(10, 5), title='nDCG@10 by Query Bucket and Method')\n",
                    "plt.ylabel('nDCG@10')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "pivot = df.pivot(index='bucket', columns='method', values='precision_at_10')\n",
                    "pivot.plot(kind='bar', figsize=(10, 5), title='Precision@10 by Query Bucket and Method')\n",
                    "plt.ylabel('Precision@10')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    questions_df = pd.read_parquet((project_root / args.questions_path).resolve())
    per_query_df = pd.read_parquet((project_root / args.per_query_metrics_path).resolve())
    overrides = load_overrides((project_root / args.overrides_path).resolve())

    bucketed_questions = add_buckets(questions_df, overrides)
    merged_df = per_query_df.merge(
        bucketed_questions[["query_id", "query_text", "query_type", "bucket", "bucket_rule_based"]],
        on=["query_id", "query_text", "query_type"],
        how="left",
    )

    merged_df["bucket"] = merged_df["bucket"].fillna("broad_exploratory")
    merged_df["bucket_rule_based"] = merged_df["bucket_rule_based"].fillna("broad_exploratory")

    bucket_metrics_df = build_bucket_metrics(merged_df)
    bucket_summary = build_bucket_summary(merged_df, bucket_metrics_df)

    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_df.to_parquet(output_dir / "per_query_with_buckets.parquet", index=False)
    bucket_metrics_df.to_csv(output_dir / "per_bucket_metrics.csv", index=False)
    (output_dir / "bucket_summary.json").write_text(
        json.dumps(bucket_summary, indent=2),
        encoding="utf-8",
    )

    notebook_path = project_root / "notebooks" / "05_query_type_analysis.ipynb"
    ensure_notebook(notebook_path)

    print("Stage 6 query-type analysis complete")
    print(f"Bucket summary: {output_dir / 'bucket_summary.json'}")
    print(f"Per-bucket metrics: {output_dir / 'per_bucket_metrics.csv'}")
    print(f"Notebook: {notebook_path}")


if __name__ == "__main__":
    main()
