# BioSeek

BioSeek is a biomedical text mining and information retrieval project built around a simple research question: how does retrieval method choice change not only relevance, but also the diversity and structure of the biomedical evidence returned for a query?

This repository brings together the project notebooks, analysis scripts, backend and frontend demo code, and supporting documentation for the final deliverable.

## Start Here

- Final deliverable notebook: `main_notebook.ipynb`
- Checkpoint 1 notebook: `checkpoints/637004205_ProjectCheckpoint1_CS676.ipynb`
- Checkpoint 2 notebook: `checkpoints/637004205_Project_Checkpoint2_CS676.ipynb`
- Supporting analysis notebooks: `notebooks/`

## Project Overview

Biomedical retrieval is typically evaluated using relevance metrics such as Precision@k, Recall, MRR, and nDCG. However, these metrics do not fully capture how different systems present and organize information. This project asks a broader question:

> How do different retrieval methods affect both the relevance and the diversity or structure of biomedical information retrieved from PubMed?

To explore this, BioSeek compares four retrieval strategies over a PMID-aligned biomedical corpus:

- TF-IDF
- BM25
- Dense semantic retrieval
- Hybrid retrieval

Beyond standard retrieval quality, the project also analyzes how these methods differ in:

- result overlap
- topical clustering
- diversity of evidence surfaced
- behavior across query types

## Research Questions

- Which retrieval method performs best on biomedical queries under standard IR metrics?
- Do higher relevance scores also correspond to broader and more diverse evidence coverage?
- How do lexical, dense, and hybrid retrieval methods behave differently across query types?

## Dataset And Processing

The project uses:

- BioASQ Task B questions and relevance judgments
- a curated PubMed subset with PMID identifiers
- exact PMID joins to align queries, relevance labels, and retrieved documents

At a high level, the workflow is:

1. Load BioASQ Task B questions and expert relevance labels.
2. Build or prepare a curated PubMed subset.
3. Join BioASQ and PubMed records by exact PMID.
4. Build retrieval indexes for TF-IDF, BM25, dense, and hybrid methods.
5. Evaluate retrieval quality across methods.
6. Run additional analyses on diversity, clustering, and query types.

## Main Results Summary

The core takeaway of BioSeek is that retrieval quality in biomedical IR should not be judged by relevance alone. Different retrieval methods can surface meaningfully different slices of the literature, which changes both the breadth of evidence and the structure of the returned result set.

The detailed quantitative results, plots, and conclusions are presented in the final curated notebook.

## Project Video

Add your final project video link here before submission.

Example:

`Project video: https://...`

## Repository Structure

```text
.
├── README.md
├── DEMO_NOTES.md
├── REPORT_NOTES.md
├── backend/
│   ├── app/
│   ├── requirements.txt
│   └── tests/
├── checkpoints/
│   ├── 637004205_ProjectCheckpoint1_CS676.ipynb
│   └── 637004205_Project_Checkpoint2_CS676.ipynb
├── docs/
│   ├── architecture.md
│   ├── clustering.md
│   ├── dense_adaptation.md
│   └── project_story.md
├── frontend/
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── public/
│   └── package.json
├── notebooks/
│   ├── 05_query_type_analysis.ipynb
│   ├── 06_diversity_analysis_dense.ipynb
│   └── README.md
├── scripts/
│   ├── analyze_diversity.py
│   ├── analyze_query_types.py
│   ├── build_retrieval_indexes.py
│   ├── cluster_retrieval_results.py
│   ├── compare_adapted_dense.py
│   ├── evaluate_retrieval.py
│   ├── join_pubmed_bioasq.py
│   ├── load_bioasq_taskb.py
│   ├── prepare_pubmed_subset.py
│   ├── run_dense_clustering_debug.py
│   ├── run_test_queries.py
│   └── train_adapted_dense.py
└── shared/
    └── config/
```

## What Each Folder Contains

- `checkpoints/`: the two checkpoint notebooks submitted earlier in the semester
- `notebooks/`: supporting exploratory and analysis notebooks used during development
- `scripts/`: data preparation, retrieval, evaluation, and analysis scripts
- `backend/`: FastAPI backend for search, comparison, analysis, and dataset endpoints
- `frontend/`: Next.js frontend for demonstrating the search and comparison workflow
- `docs/`: supporting writeups on architecture, clustering, dense retrieval, and project framing
- `shared/config/`: shared configuration used across backend and frontend pieces

## How To Reproduce The Work

This project was developed primarily in Colab for notebook-based experimentation, with a local frontend and backend for the interactive demo.

### Notebook Reproduction

1. Open `main_notebook.ipynb`.
2. Install the dependencies listed in `requirements.txt`.
3. Run the notebook cells in order.
4. Use the supporting scripts in `scripts/` if you need to rebuild processed data or retrieval indexes.

If your final submission still relies on Colab, export a `requirements.txt` from Colab and place it at the repository root before submitting.

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
npm run dev
```

## Key Dependencies And Versions

Important Python dependencies currently tracked in `backend/requirements.txt` include:

- `fastapi==0.115.6`
- `uvicorn[standard]==0.32.1`
- `scikit-learn==1.6.0`
- `sentence-transformers==3.3.1`
- `faiss-cpu==1.13.2`
- `pandas==3.0.2`
- `pyarrow==23.0.1`
- `datasets==3.2.0`

Important frontend dependencies currently tracked in `frontend/package.json` include:

- `next==14.2.15`
- `react==18.3.1`
- `react-dom==18.3.1`
- `tailwindcss==3.4.17`
- `typescript==5.7.2`

Before submission, also list the Python version used for the final notebook near the top of this README.

## Main Scripts

- `scripts/load_bioasq_taskb.py`: loads BioASQ Task B questions and labels
- `scripts/prepare_pubmed_subset.py`: prepares the curated PubMed subset
- `scripts/join_pubmed_bioasq.py`: joins BioASQ and PubMed by PMID
- `scripts/build_retrieval_indexes.py`: builds retrieval indexes
- `scripts/evaluate_retrieval.py`: computes retrieval evaluation metrics
- `scripts/analyze_query_types.py`: analyzes retrieval behavior by query type
- `scripts/analyze_diversity.py`: analyzes diversity across methods
- `scripts/cluster_retrieval_results.py`: clusters retrieved results for structural analysis

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

## Supporting Documents

- [Project story](docs/project_story.md)
- [Architecture notes](docs/architecture.md)
- [Clustering notes](docs/clustering.md)
- [Dense adaptation notes](docs/dense_adaptation.md)
- [Demo notes](DEMO_NOTES.md)
- [Report notes](REPORT_NOTES.md)

## Final Submission Checklist

Before submitting the public GitHub link, make sure this repository includes:

- `main_notebook.ipynb` at the expected filename
- both checkpoint notebooks in `checkpoints/`
- a root-level `requirements.txt` exported from Colab
- a project video link in this README
- a short Python version note in this README
- any required data access instructions if raw data is too large to commit

## Notes

The current repository already contains the main scripts, checkpoint notebooks, backend, frontend, and supporting documentation. The two remaining deliverable-specific items that should be added before submission are:

- the final curated notebook named `main_notebook.ipynb`
- the root `requirements.txt` exported from the notebook environment
