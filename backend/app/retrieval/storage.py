from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
INDEX_ROOT = DATA_ROOT / "indexes"

SUPPORTED_DATASETS = {
    "pubmed_subset": PROCESSED_ROOT / "pubmed_subset" / "docs.parquet",
    "unified": PROCESSED_ROOT / "unified" / "docs.parquet",
}


def get_dataset_docs_path(dataset_name: str) -> Path:
    try:
        return SUPPORTED_DATASETS[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Expected one of: {sorted(SUPPORTED_DATASETS)}"
        ) from exc


def load_dataset_docs(dataset_name: str) -> pd.DataFrame:
    docs_path = get_dataset_docs_path(dataset_name)
    if not docs_path.exists():
        raise FileNotFoundError(f"Dataset docs parquet not found: {docs_path}")

    docs_df = pd.read_parquet(docs_path)
    docs_df["pmid"] = docs_df["pmid"].astype("string")
    docs_df["title"] = docs_df["title"].astype("string")
    docs_df["abstract"] = docs_df["abstract"].astype("string")
    docs_df["journal"] = docs_df["journal"].astype("string")
    docs_df["retrieval_text"] = docs_df["retrieval_text"].astype("string")
    return docs_df


def get_method_index_dir(dataset_name: str, method_name: str) -> Path:
    index_dir = INDEX_ROOT / dataset_name / method_name
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_sparse_matrix(path: Path, matrix: sparse.spmatrix) -> None:
    sparse.save_npz(path, matrix)


def load_sparse_matrix(path: Path) -> sparse.spmatrix:
    return sparse.load_npz(path)

