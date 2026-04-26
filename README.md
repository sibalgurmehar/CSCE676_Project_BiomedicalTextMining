# BioSeek

BioSeek is a biomedical information retrieval and data mining course project. It studies how retrieval method choice changes both relevance and the structure or diversity of PubMed evidence returned for biomedical questions.

## Research Question

How do different retrieval methods affect both the relevance and the diversity or structure of biomedical information retrieved from PubMed?

## Final Project Story

The final project pipeline is:

1. Load BioASQ Task B questions and expert-labeled relevant PubMed IDs.
2. Build a curated PubMed subset with PMIDs, keeping mapped BioASQ documents and additional distractors.
3. Join BioASQ and PubMed using exact PMIDs only.
4. Build four retrieval methods over the resulting corpora:
   - TF-IDF
   - BM25
   - Dense semantic retrieval
   - Hybrid retrieval
5. Evaluate retrieval on the unified PubMed-BioASQ dataset.
6. Extend the analysis with query-type analysis, clustering, and diversity analysis.

## Data Design

- Main document source: curated PubMed subset with PMIDs
- Query and relevance source: BioASQ Task B
- Unified evaluation dataset: BioASQ questions and qrels joined to curated PubMed docs by exact PMID
- Exploration corpus: curated PubMed subset for live search and clustering

## Implemented Analysis

- Retrieval evaluation:
  - Precision@5
  - Precision@10
  - Recall@10
  - MRR
  - nDCG@5
  - nDCG@10
- Query-type analysis
- Clustering of retrieved PubMed documents
- Diversity analysis across retrieval methods
- Optional AI-assisted polish layer for rewrites, cluster refinement, and retrieval explanations

## Stack

- Frontend: Next.js 14, TypeScript, Tailwind CSS
- Backend: FastAPI, Python
- Lexical retrieval: scikit-learn TF-IDF and BM25
- Dense retrieval: sentence-transformers
- Vector index: FAISS
- Data formats: Parquet, CSV, JSON

## Repository Layout

```text
frontend/
backend/
scripts/
data/
notebooks/
eval/
docs/
shared/
```

## Run Locally

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api npm run dev
```

## Main API Routes

- `GET /api/health`
- `POST /api/search`
- `POST /api/compare`
- `POST /api/clusters`
- `POST /api/query-analysis`
- `GET /api/metrics/summary`
- `GET /api/query-types/summary`
- `GET /api/metrics/examples`
- `GET /api/diversity/summary`
- `GET /api/dataset-info`
- `POST /api/ai-polish`

## Documents

- [Architecture](docs/architecture.md)
- [Project Story](docs/project_story.md)
- [Clustering Notes](docs/clustering.md)
- [Dense Adaptation Notes](docs/dense_adaptation.md)
- [Demo Notes](DEMO_NOTES.md)
- [Report Notes](REPORT_NOTES.md)

## Important Constraint

Core evaluation is based on PubMed documents with BioASQ expert relevance annotations joined by exact PMID. The optional AI-assisted layer does not affect retrieval metrics or benchmark results.
