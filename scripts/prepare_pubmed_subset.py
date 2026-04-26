import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a curated PubMed subset with BioASQ-relevant PMIDs and distractors."
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=100_000,
        help="Maximum number of documents to keep in the curated subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reserved for reproducibility metadata.",
    )
    parser.add_argument(
        "--relevance-path",
        type=Path,
        default=Path("data/processed/bioasq/relevance.parquet"),
        help="Stage 1 relevance parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/pubmed_subset"),
        help="Directory for the curated PubMed subset outputs.",
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=None,
        help="Optional directory of official PubMed XML files for fallback parsing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="Print progress after this many scanned records.",
    )
    return parser.parse_args()


def load_required_pmids(relevance_path: Path) -> set[str]:
    relevance_df = pd.read_parquet(relevance_path, columns=["pmid"])
    pmids = relevance_df["pmid"].astype("string").dropna().str.strip()
    return set(pmids[pmids != ""])


def stream_huggingface_pubmed() -> Iterable[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "The 'huggingface-hub' package is required for Hugging Face PubMed streaming."
        ) from exc

    parquet_path = hf_hub_download(
        repo_id="jmhb/pubmed_bioasq_2022",
        repo_type="dataset",
        filename="data/allMeSH_2022.parquet",
    )

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("The 'pyarrow' package is required for parquet batch streaming.") from exc

    parquet_file = pq.ParquetFile(parquet_path)
    columns = ["pmid", "title", "abstractText", "meshMajor", "journal", "year"]

    for batch in parquet_file.iter_batches(columns=columns, batch_size=10_000):
        frame = batch.to_pandas()
        for record in frame.to_dict(orient="records"):
            yield record


def stream_pubmed_xml(xml_dir: Path) -> Iterable[dict[str, Any]]:
    try:
        from lxml import etree
    except ImportError as exc:  # pragma: no cover - environment-dependent fallback
        raise RuntimeError("The 'lxml' package is required for PubMed XML fallback parsing.") from exc

    for xml_path in sorted(xml_dir.glob("*.xml*")):
        context = etree.iterparse(str(xml_path), events=("end",), tag="PubmedArticle")
        for _, article in context:
            pmid = article.findtext(".//MedlineCitation/PMID")
            title = article.findtext(".//ArticleTitle")
            abstract_parts = article.findall(".//Abstract/AbstractText")
            abstract = " ".join("".join(part.itertext()).strip() for part in abstract_parts if "".join(part.itertext()).strip())
            journal = article.findtext(".//Journal/Title")

            year_text = (
                article.findtext(".//PubDate/Year")
                or article.findtext(".//ArticleDate/Year")
                or article.findtext(".//DateCompleted/Year")
            )

            mesh_terms = []
            for mesh_heading in article.findall(".//MeshHeading/DescriptorName"):
                text = "".join(mesh_heading.itertext()).strip()
                if text:
                    mesh_terms.append(text)

            yield {
                "pmid": pmid,
                "title": title,
                "abstractText": abstract,
                "meshMajor": mesh_terms,
                "journal": journal,
                "year": int(year_text) if year_text and year_text.isdigit() else None,
            }

            article.clear()
            while article.getprevious() is not None:
                del article.getparent()[0]


def as_clean_string(value: Any) -> str:
    if value is None:
        return ""

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, list):
        return " ".join(as_clean_string(item) for item in value if as_clean_string(item)).strip()

    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_mesh_terms(value: Any) -> list[str]:
    if value is None:
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, list):
        terms = [as_clean_string(item) for item in value]
        return [term for term in terms if term]

    if isinstance(value, dict):
        terms = []
        for key in ("meshMajor", "meshMinor", "terms", "labels"):
            nested = value.get(key)
            if nested:
                terms.extend(as_mesh_terms(nested))
        return terms

    text = as_clean_string(value)
    return [text] if text else []


def first_present_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue

        if hasattr(value, "tolist"):
            converted = value.tolist()
            if converted is None:
                continue
            if isinstance(converted, list) and len(converted) == 0:
                continue
            return value

        if isinstance(value, list) and len(value) == 0:
            continue

        if isinstance(value, str) and value.strip() == "":
            continue

        return value

    return None


def normalize_pubmed_record(record: dict[str, Any]) -> dict[str, Any] | None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}

    pmid = as_clean_string(
        first_present_value(
            record.get("pmid"),
            record.get("_id"),
            metadata.get("pmid"),
            metadata.get("_id"),
        )
    )
    if not pmid:
        return None

    title = as_clean_string(first_present_value(record.get("title"), metadata.get("title")))
    abstract = as_clean_string(
        first_present_value(
            record.get("abstractText"),
            record.get("abstract"),
            record.get("text"),
            metadata.get("abstractText"),
            metadata.get("abstract"),
        )
    )
    journal = as_clean_string(first_present_value(record.get("journal"), metadata.get("journal")))
    year = as_int(first_present_value(record.get("year"), metadata.get("year")))
    mesh_terms = as_mesh_terms(
        first_present_value(
            record.get("meshMajor"),
            record.get("mesh_terms"),
            metadata.get("meshMajor"),
            metadata.get("mesh_terms"),
        )
    )

    retrieval_text = f"{title} {abstract}".strip() if abstract else title

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "mesh_terms": mesh_terms,
        "journal": journal,
        "year": year,
        "retrieval_text": retrieval_text,
    }


def build_subset(
    source_iterable: Iterable[dict[str, Any]],
    required_pmids: set[str],
    target_size: int,
    seed: int,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(required_pmids) > target_size:
        raise ValueError(
            f"Target size {target_size} is too small for {len(required_pmids)} required BioASQ PMIDs."
        )

    distractor_target = target_size - len(required_pmids)

    relevant_docs: dict[str, dict[str, Any]] = {}
    distractors: list[dict[str, Any]] = []
    selected_distractor_pmids: set[str] = set()
    scanned_records = 0
    normalized_records = 0
    invalid_records = 0
    raw_record_samples: list[dict[str, Any]] = []

    for record in source_iterable:
        scanned_records += 1
        if len(raw_record_samples) < 3 and isinstance(record, dict):
            raw_record_samples.append(
                {
                    "keys": sorted(str(key) for key in record.keys())[:20],
                    "preview": json.dumps(record, default=str)[:500],
                }
            )

        normalized = normalize_pubmed_record(record)
        if normalized is None:
            invalid_records += 1
            if scanned_records >= 1000 and normalized_records == 0:
                raise RuntimeError(
                    "Could not normalize any of the first 1000 streamed PubMed records. "
                    f"Sample raw records: {json.dumps(raw_record_samples, indent=2)}"
                )
            if progress_every > 0 and scanned_records % progress_every == 0:
                print_progress(
                    scanned_records=scanned_records,
                    required_found=len(relevant_docs),
                    required_total=len(required_pmids),
                    distractors_kept=len(distractors),
                    target_size=target_size,
                )
            continue

        normalized_records += 1
        pmid = normalized["pmid"]
        if pmid in required_pmids:
            relevant_docs.setdefault(pmid, normalized)
            if progress_every > 0 and scanned_records % progress_every == 0:
                print_progress(
                    scanned_records=scanned_records,
                    required_found=len(relevant_docs),
                    required_total=len(required_pmids),
                    distractors_kept=len(distractors),
                    target_size=target_size,
                )
            if len(relevant_docs) == len(required_pmids) and len(distractors) >= distractor_target:
                break
            continue

        if pmid in selected_distractor_pmids:
            if progress_every > 0 and scanned_records % progress_every == 0:
                print_progress(
                    scanned_records=scanned_records,
                    required_found=len(relevant_docs),
                    required_total=len(required_pmids),
                    distractors_kept=len(distractors),
                    target_size=target_size,
                )
            continue

        if len(distractors) < distractor_target:
            distractors.append(normalized)
            selected_distractor_pmids.add(pmid)

        if progress_every > 0 and scanned_records % progress_every == 0:
            print_progress(
                scanned_records=scanned_records,
                required_found=len(relevant_docs),
                required_total=len(required_pmids),
                distractors_kept=len(distractors),
                target_size=target_size,
            )

        if len(relevant_docs) == len(required_pmids) and len(distractors) >= distractor_target:
            break

    final_docs = list(relevant_docs.values())
    distractor_rows = [doc for doc in distractors if doc["pmid"] not in required_pmids]
    available_slots = max(target_size - len(final_docs), 0)
    final_docs.extend(distractor_rows[:available_slots])

    docs_df = pd.DataFrame(
        final_docs,
        columns=["pmid", "title", "abstract", "mesh_terms", "journal", "year", "retrieval_text"],
    )
    docs_df = docs_df.drop_duplicates(subset=["pmid"]).reset_index(drop=True)
    docs_df["pmid"] = docs_df["pmid"].astype("string")
    docs_df["title"] = docs_df["title"].astype("string")
    docs_df["abstract"] = docs_df["abstract"].astype("string")
    docs_df["journal"] = docs_df["journal"].astype("string")
    docs_df["retrieval_text"] = docs_df["retrieval_text"].astype("string")
    docs_df["year"] = docs_df["year"].astype("Int64")

    stats = {
        "total_docs": int(len(docs_df)),
        "docs_with_abstract": int(docs_df["abstract"].fillna("").str.strip().ne("").sum()),
        "docs_with_mesh": int(docs_df["mesh_terms"].apply(lambda terms: len(terms) > 0).sum()),
        "bioasq_pmids_required": int(len(required_pmids)),
        "bioasq_pmids_found": int(len(relevant_docs)),
        "bioasq_pmids_missing": int(len(required_pmids) - len(relevant_docs)),
        "distractors_added": int(max(len(docs_df) - len(relevant_docs), 0)),
        "scanned_records": int(scanned_records),
        "normalized_records": int(normalized_records),
        "invalid_records": int(invalid_records),
        "target_size": int(target_size),
        "seed": int(seed),
    }

    return docs_df, stats


def print_progress(
    scanned_records: int,
    required_found: int,
    required_total: int,
    distractors_kept: int,
    target_size: int,
) -> None:
    print(
        "Progress:"
        f" scanned={scanned_records:,}"
        f" required_found={required_found:,}/{required_total:,}"
        f" distractors_kept={distractors_kept:,}"
        f" current_total={required_found + distractors_kept:,}/{target_size:,}",
        flush=True,
    )


def save_outputs(
    docs_df: pd.DataFrame,
    stats: dict[str, Any],
    output_dir: Path,
    source_report: dict[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    docs_path = output_dir / "docs.parquet"
    stats_path = output_dir / "stats.json"
    source_path = output_dir / "source_report.json"

    docs_df.to_parquet(docs_path, index=False)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    source_path.write_text(json.dumps(source_report, indent=2), encoding="utf-8")

    return docs_path, stats_path, source_path


def validate_output(docs_path: Path) -> pd.DataFrame:
    reloaded = pd.read_parquet(docs_path)
    if "pmid" not in reloaded.columns:
        raise AssertionError("Output parquet is missing the 'pmid' column.")
    if reloaded["pmid"].isna().any():
        raise AssertionError("Output parquet contains missing PMIDs.")
    return reloaded


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    relevance_path = (project_root / args.relevance_path).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    required_pmids = load_required_pmids(relevance_path)

    source_iterable: Iterable[dict[str, Any]]
    source_report: dict[str, Any]

    try:
        source_iterable = stream_huggingface_pubmed()
        source_report = {
            "source_type": "huggingface_parquet_batches",
            "dataset": "jmhb/pubmed_bioasq_2022",
            "file": "data/allMeSH_2022.parquet",
            "batch_size": 10000,
            "streaming": False,
            "batch_streaming": True,
            "reference": "https://huggingface.co/datasets/jmhb/pubmed_bioasq_2022",
            "required_pmids_path": str(relevance_path),
        }
    except Exception as exc:
        if args.xml_dir is None:
            raise RuntimeError(
                "Failed to initialize Hugging Face streaming dataset and no XML fallback directory was provided."
            ) from exc

        xml_dir = (project_root / args.xml_dir).resolve()
        source_iterable = stream_pubmed_xml(xml_dir)
        source_report = {
            "source_type": "pubmed_xml_fallback",
            "xml_dir": str(xml_dir),
            "reference": "https://pubmed.ncbi.nlm.nih.gov/download/",
            "required_pmids_path": str(relevance_path),
        }

    docs_df, stats = build_subset(
        source_iterable=source_iterable,
        required_pmids=required_pmids,
        target_size=args.target_size,
        seed=args.seed,
        progress_every=args.progress_every,
    )

    docs_path, stats_path, source_path = save_outputs(
        docs_df=docs_df,
        stats=stats,
        output_dir=output_dir,
        source_report=source_report,
    )

    reloaded = validate_output(docs_path)

    print("PubMed Stage 2 complete")
    print(f"Rows saved: {len(reloaded)}")
    print(f"BioASQ PMIDs found: {stats['bioasq_pmids_found']} / {stats['bioasq_pmids_required']}")
    print(f"Distractors added: {stats['distractors_added']}")
    print(f"Docs with abstract: {stats['docs_with_abstract']}")
    print(f"Docs with mesh: {stats['docs_with_mesh']}")
    print(f"Docs parquet: {docs_path}")
    print(f"Stats JSON: {stats_path}")
    print(f"Source report: {source_path}")
    print("Sample rows:")
    print(reloaded.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
