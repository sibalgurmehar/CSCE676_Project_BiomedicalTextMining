import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load BioASQ Task B questions and relevant PubMed IDs."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Path to a manually downloaded BioASQ Task B JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/bioasq"),
        help="Directory for normalized parquet outputs.",
    )
    return parser.parse_args()


def find_local_json(project_root: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None

    candidate_patterns = [
        "data/raw/bioasq/**/*.json",
        "data/raw/**/*.json",
        "data/**/*.json",
    ]

    for pattern in candidate_patterns:
        for candidate in sorted(project_root.glob(pattern)):
            if candidate.is_file() and "bioasq" in candidate.name.lower():
                return candidate
    return None


def load_questions_from_local_json(json_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        return payload["questions"]

    raise ValueError(
        f"Unsupported BioASQ JSON structure in {json_path}. Expected a top-level 'questions' list."
    )


def extract_questions_from_hf_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id") or item.get("question_id") or item.get("query_id"),
        "body": item.get("body") or item.get("question") or item.get("query"),
        "type": item.get("type") or item.get("question_type"),
        "documents": item.get("documents") or item.get("relevant_documents") or [],
        "snippets": item.get("snippets") or [],
    }


def load_questions_from_hf_parquet() -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "The 'huggingface-hub' package is required for direct Hugging Face parquet loading."
        ) from exc

    api = HfApi()
    repo_files = api.list_repo_files(repo_id="jmhb/BioASQ", repo_type="dataset")
    parquet_files = sorted(
        file_name for file_name in repo_files if file_name.startswith("data/") and file_name.endswith(".parquet")
    )

    if not parquet_files:
        raise RuntimeError("No parquet files were found in the Hugging Face dataset repo.")

    frames: list[pd.DataFrame] = []
    for parquet_file in parquet_files:
        local_path = hf_hub_download(
            repo_id="jmhb/BioASQ",
            repo_type="dataset",
            filename=parquet_file,
        )
        frame = pd.read_parquet(local_path)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    rename_map = {}
    if "question" in combined.columns and "body" not in combined.columns:
        rename_map["question"] = "body"
    if rename_map:
        combined = combined.rename(columns=rename_map)

    expected_columns = ["id", "body", "type", "documents", "snippets"]
    for column in expected_columns:
        if column not in combined.columns:
            combined[column] = None if column != "documents" and column != "snippets" else []

    return combined[expected_columns].to_dict(orient="records")


def load_questions_from_huggingface() -> list[dict[str, Any]]:
    parquet_error: Exception | None = None

    try:
        return load_questions_from_hf_parquet()
    except Exception as exc:  # pragma: no cover - fallback path
        parquet_error = exc

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for Hugging Face fallback loading."
        ) from exc

    try:
        dataset = load_dataset("jmhb/BioASQ")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Hugging Face BioASQ data via direct parquet download and via datasets.load_dataset."
        ) from (parquet_error or exc)

    if "train" in dataset:
        split = dataset["train"]
    else:
        first_split = next(iter(dataset.keys()))
        split = dataset[first_split]

    return [extract_questions_from_hf_item(dict(row)) for row in split]


def extract_pmid(document_ref: Any) -> str | None:
    if document_ref is None:
        return None

    if isinstance(document_ref, dict):
        for key in ("pmid", "id", "document"):
            value = document_ref.get(key)
            if value:
                return str(value).strip()
        return None

    raw = str(document_ref).strip()
    if not raw:
        return None

    if raw.startswith("http://") or raw.startswith("https://"):
        return raw.rstrip("/").split("/")[-1]

    return raw


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        if converted is None:
            return []
        return [converted]

    if isinstance(value, float) and math.isnan(value):
        return []

    if pd.isna(value):
        return []

    return [value]


def normalize_questions(raw_questions: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    question_rows: list[dict[str, Any]] = []
    relevance_rows: list[dict[str, Any]] = []
    snippet_count = 0

    for item in raw_questions:
        query_id = item.get("id") or item.get("question_id") or item.get("query_id")
        query_text = item.get("body") or item.get("question") or item.get("query")
        query_type = item.get("type") or item.get("question_type")
        snippets = ensure_list(item.get("snippets"))
        documents = ensure_list(item.get("documents"))
        if not documents:
            documents = ensure_list(item.get("relevant_documents"))

        if not query_id or not query_text:
            continue

        query_id = str(query_id).strip()
        query_text = str(query_text).strip()
        query_type = str(query_type).strip() if query_type is not None else None

        question_rows.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "query_type": query_type,
            }
        )

        snippet_count += len(snippets)

        seen_pmids: set[str] = set()
        for document_ref in documents:
            pmid = extract_pmid(document_ref)
            if pmid is None or pmid in seen_pmids:
                continue

            seen_pmids.add(pmid)
            relevance_rows.append(
                {
                    "query_id": query_id,
                    "pmid": pmid,
                    "relevance": 1,
                }
            )

    questions_df = pd.DataFrame(
        question_rows, columns=["query_id", "query_text", "query_type"]
    )
    relevance_df = pd.DataFrame(
        relevance_rows, columns=["query_id", "pmid", "relevance"]
    )

    if not questions_df.empty:
        questions_df = (
            questions_df.drop_duplicates(subset=["query_id"])
            .sort_values("query_id")
            .reset_index(drop=True)
        )
    questions_df = questions_df.astype(
        {"query_id": "string", "query_text": "string", "query_type": "string"}
    )

    if not relevance_df.empty:
        relevance_df = (
            relevance_df.drop_duplicates(subset=["query_id", "pmid"])
            .astype({"query_id": "string", "pmid": "string", "relevance": "int64"})
            .sort_values(["query_id", "pmid"])
            .reset_index(drop=True)
        )

    avg_relevant_docs = (
        float(relevance_df.groupby("query_id")["pmid"].count().mean())
        if not relevance_df.empty
        else 0.0
    )

    stats = {
        "num_questions": int(len(questions_df)),
        "num_unique_pmids": int(relevance_df["pmid"].nunique()) if not relevance_df.empty else 0,
        "avg_relevant_docs_per_question": round(avg_relevant_docs, 4),
        "num_relevance_rows": int(len(relevance_df)),
        "num_snippets": int(snippet_count),
    }

    return questions_df, relevance_df, stats


def save_outputs(
    output_dir: Path,
    questions_df: pd.DataFrame,
    relevance_df: pd.DataFrame,
    stats: dict[str, Any],
    source: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    questions_path = output_dir / "questions.parquet"
    relevance_path = output_dir / "relevance.parquet"
    stats_path = output_dir / "stats.json"

    questions_df.to_parquet(questions_path, index=False)
    questions_df.to_csv(output_dir / "questions.csv", index=False)

    relevance_df.to_parquet(relevance_path, index=False)
    relevance_df.to_csv(output_dir / "relevance.csv", index=False)

    stats_payload = {
        "source": source,
        "outputs": {
            "questions_parquet": str(questions_path),
            "relevance_parquet": str(relevance_path),
        },
        **stats,
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = (project_root / args.output_dir).resolve()

    local_json = find_local_json(project_root, args.input_json)

    if local_json is not None:
        raw_questions = load_questions_from_local_json(local_json)
        source = {"type": "local_json", "path": str(local_json.resolve())}
    else:
        raw_questions = load_questions_from_huggingface()
        source = {"type": "huggingface", "dataset": "jmhb/BioASQ"}

    questions_df, relevance_df, stats = normalize_questions(raw_questions)
    save_outputs(output_dir, questions_df, relevance_df, stats, source)

    print("BioASQ Stage 1 complete")
    print(f"Questions: {stats['num_questions']}")
    print(f"Unique PMIDs: {stats['num_unique_pmids']}")
    print(f"Average relevant docs per question: {stats['avg_relevant_docs_per_question']}")
    print(f"Output directory: {output_dir}")
    print("PMIDs were preserved as strings for exact Stage 3 joining.")


if __name__ == "__main__":
    main()
