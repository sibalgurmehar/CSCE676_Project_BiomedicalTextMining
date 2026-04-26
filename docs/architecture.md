# BioSeek Architecture

## 1. Project goal

BioSeek is a student-friendly full-stack system for comparing biomedical retrieval methods on a unified PubMed and BioASQ dataset. The system is designed to support both interactive exploration and course-style evaluation.

## 2. Main workflow

1. Build a curated PubMed subset with PMIDs.
2. Collect BioASQ Task B questions and relevance labels.
3. Join BioASQ labels to PubMed documents by exact PMID.
4. Build multiple retrieval indexes over the same document collection.
5. Run retrieval for each query and compare results.
6. Analyze relevance, diversity, clustering behavior, and query-type differences.

## 3. High-level architecture

### Data layer

- `data/raw/`: original PubMed and BioASQ files
- `data/processed/`: cleaned joins, evaluation tables, embedding-ready corpora
- Shared document key: PMID

### Retrieval layer

- TF-IDF index using `scikit-learn`
- BM25 index using a lightweight Python BM25 implementation
- Dense semantic retrieval using `sentence-transformers`
- FAISS vector index for dense retrieval
- Hybrid retrieval by combining lexical and dense scores

### Analysis layer

- Relevance metrics such as Precision@k, Recall@k, MAP, and nDCG
- Diversity analysis across retrieval methods
- Clustering of retrieved documents
- Query-type analysis for different biomedical question styles

### Backend layer

- FastAPI service in `backend/`
- Exposes search, configuration, and analysis endpoints
- Starts with mock data and can later call real retrieval modules

### Frontend layer

- Next.js 14 app in `frontend/`
- Dashboard-style UI for methods, metrics, and example retrieval outputs
- Uses mock JSON during early development

## 4. Shared configuration

`shared/config/settings.json` is the single source of truth for:

- project name and research question
- retrieval methods
- analysis modules
- planned datasets
- default frontend display settings

Frontend and backend each load this file through a local wrapper module.

## 5. Design principles

- Keep the system understandable for a course project
- Prefer transparency over engineering cleverness
- Separate data preparation, retrieval, evaluation, and UI clearly
- Make each method comparison easy to inspect and explain

## 6. Next implementation milestones

1. Add data ingestion for PubMed subset and BioASQ Task B.
2. Implement exact PMID joining and validation.
3. Build retrieval indexes for all four methods.
4. Add evaluation scripts under `eval/`.
5. Connect frontend visualizations to live backend outputs.

