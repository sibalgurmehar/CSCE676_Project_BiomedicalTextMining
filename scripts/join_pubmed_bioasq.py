import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the unified BioSeek dataset by exact PMID join."
    )
    parser.add_argument(
        "--docs-path",
        type=Path,
        default=Path("data/processed/pubmed_subset/docs.parquet"),
        help="Path to the curated PubMed subset parquet.",
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("data/processed/bioasq/questions.parquet"),
        help="Path to the BioASQ questions parquet.",
    )
    parser.add_argument(
        "--relevance-path",
        type=Path,
        default=Path("data/processed/bioasq/relevance.parquet"),
        help="Path to the BioASQ relevance parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/unified"),
        help="Directory for unified questions, docs, qrels, and mapping report.",
    )
    return parser.parse_args()


def clean_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def load_inputs(
    docs_path: Path,
    questions_path: Path,
    relevance_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    docs_df = pd.read_parquet(docs_path)
    questions_df = pd.read_parquet(questions_path)
    relevance_df = pd.read_parquet(relevance_path)

    required_doc_columns = ["pmid", "title", "abstract", "mesh_terms", "journal", "year", "retrieval_text"]
    required_question_columns = ["query_id", "query_text", "query_type"]
    required_relevance_columns = ["query_id", "pmid", "relevance"]

    missing_docs = [column for column in required_doc_columns if column not in docs_df.columns]
    missing_questions = [column for column in required_question_columns if column not in questions_df.columns]
    missing_relevance = [column for column in required_relevance_columns if column not in relevance_df.columns]

    if missing_docs:
        raise ValueError(f"Docs parquet is missing required columns: {missing_docs}")
    if missing_questions:
        raise ValueError(f"Questions parquet is missing required columns: {missing_questions}")
    if missing_relevance:
        raise ValueError(f"Relevance parquet is missing required columns: {missing_relevance}")

    docs_df = docs_df[required_doc_columns].copy()
    questions_df = questions_df[required_question_columns].copy()
    relevance_df = relevance_df[required_relevance_columns].copy()

    docs_df["pmid"] = clean_string_series(docs_df["pmid"])
    questions_df["query_id"] = clean_string_series(questions_df["query_id"])
    questions_df["query_text"] = questions_df["query_text"].astype("string")
    questions_df["query_type"] = questions_df["query_type"].astype("string")
    relevance_df["query_id"] = clean_string_series(relevance_df["query_id"])
    relevance_df["pmid"] = clean_string_series(relevance_df["pmid"])
    relevance_df["relevance"] = relevance_df["relevance"].fillna(1).astype("int64")

    docs_df = docs_df[docs_df["pmid"] != ""].drop_duplicates(subset=["pmid"]).reset_index(drop=True)
    questions_df = questions_df[questions_df["query_id"] != ""].drop_duplicates(subset=["query_id"]).reset_index(drop=True)
    relevance_df = (
        relevance_df[(relevance_df["query_id"] != "") & (relevance_df["pmid"] != "")]
        .drop_duplicates(subset=["query_id", "pmid"])
        .reset_index(drop=True)
    )

    return docs_df, questions_df, relevance_df


def build_unified_dataset(
    docs_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    relevance_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    total_bioasq_questions = int(len(questions_df))
    total_subset_docs = int(len(docs_df))
    total_bioasq_unique_pmids = int(relevance_df["pmid"].nunique()) if not relevance_df.empty else 0

    subset_pmids = set(docs_df["pmid"].tolist())
    mapped_qrels = relevance_df[relevance_df["pmid"].isin(subset_pmids)].copy()

    surviving_query_ids = set(mapped_qrels["query_id"].tolist())
    surviving_questions = (
        questions_df[questions_df["query_id"].isin(surviving_query_ids)]
        .sort_values("query_id")
        .reset_index(drop=True)
    )

    mapped_doc_pmids = set(mapped_qrels["pmid"].tolist())
    surviving_docs = (
        docs_df[docs_df["pmid"].isin(mapped_doc_pmids)]
        .sort_values("pmid")
        .reset_index(drop=True)
    )

    surviving_qrels = (
        mapped_qrels[mapped_qrels["query_id"].isin(set(surviving_questions["query_id"].tolist()))]
        .sort_values(["query_id", "pmid"])
        .reset_index(drop=True)
    )

    avg_relevant_docs = (
        float(surviving_qrels.groupby("query_id")["pmid"].count().mean())
        if not surviving_qrels.empty
        else 0.0
    )
    coverage_pct = (
        round((len(mapped_doc_pmids) / total_bioasq_unique_pmids) * 100, 4)
        if total_bioasq_unique_pmids > 0
        else 0.0
    )

    mapping_report = {
        "total_bioasq_questions": total_bioasq_questions,
        "total_docs_in_curated_pubmed_subset": total_subset_docs,
        "total_bioasq_relevance_rows": int(len(relevance_df)),
        "total_bioasq_unique_pmids": total_bioasq_unique_pmids,
        "total_mapped_relevant_docs": int(len(mapped_doc_pmids)),
        "total_mapped_qrels": int(len(surviving_qrels)),
        "number_of_surviving_queries": int(len(surviving_questions)),
        "average_relevant_docs_per_surviving_query": round(avg_relevant_docs, 4),
        "percentage_of_bioasq_relevant_pmids_found_in_corpus": coverage_pct,
        "exact_pmid_join_only": True,
        "notes": "Only exact PMID matches between BioASQ relevance labels and the curated PubMed subset were kept.",
    }

    return surviving_questions, surviving_docs, surviving_qrels, mapping_report


def save_outputs(
    output_dir: Path,
    questions_df: pd.DataFrame,
    docs_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    mapping_report: dict,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    questions_path = output_dir / "questions.parquet"
    docs_path = output_dir / "docs.parquet"
    qrels_path = output_dir / "qrels.parquet"
    report_path = output_dir / "mapping_report.json"

    questions_df.to_parquet(questions_path, index=False)
    docs_df.to_parquet(docs_path, index=False)
    qrels_df.to_parquet(qrels_path, index=False)
    report_path.write_text(json.dumps(mapping_report, indent=2), encoding="utf-8")

    return questions_path, docs_path, qrels_path, report_path


def validate_outputs(
    questions_path: Path,
    docs_path: Path,
    qrels_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    questions_df = pd.read_parquet(questions_path)
    docs_df = pd.read_parquet(docs_path)
    qrels_df = pd.read_parquet(qrels_path)

    if "query_id" not in questions_df.columns:
        raise AssertionError("Unified questions parquet is missing 'query_id'.")
    if "pmid" not in docs_df.columns:
        raise AssertionError("Unified docs parquet is missing 'pmid'.")
    if not {"query_id", "pmid", "relevance"}.issubset(qrels_df.columns):
        raise AssertionError("Unified qrels parquet is missing required columns.")

    return questions_df, docs_df, qrels_df


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    docs_path = (project_root / args.docs_path).resolve()
    questions_path = (project_root / args.questions_path).resolve()
    relevance_path = (project_root / args.relevance_path).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    docs_df, questions_df, relevance_df = load_inputs(
        docs_path=docs_path,
        questions_path=questions_path,
        relevance_path=relevance_path,
    )

    unified_questions, unified_docs, unified_qrels, mapping_report = build_unified_dataset(
        docs_df=docs_df,
        questions_df=questions_df,
        relevance_df=relevance_df,
    )

    questions_out, docs_out, qrels_out, report_out = save_outputs(
        output_dir=output_dir,
        questions_df=unified_questions,
        docs_df=unified_docs,
        qrels_df=unified_qrels,
        mapping_report=mapping_report,
    )

    reloaded_questions, reloaded_docs, reloaded_qrels = validate_outputs(
        questions_path=questions_out,
        docs_path=docs_out,
        qrels_path=qrels_out,
    )

    print("BioSeek Stage 3 complete")
    print(f"Unified questions: {len(reloaded_questions)}")
    print(f"Unified docs: {len(reloaded_docs)}")
    print(f"Unified qrels: {len(reloaded_qrels)}")
    print(
        "PMID coverage:"
        f" {mapping_report['percentage_of_bioasq_relevant_pmids_found_in_corpus']}%"
    )
    print(f"Mapping report: {report_out}")


if __name__ == "__main__":
    main()
