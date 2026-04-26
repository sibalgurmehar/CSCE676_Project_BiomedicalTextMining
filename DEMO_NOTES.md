# Demo Notes

## What To Show First

1. Search page
   - Enter a biomedical query such as `diabetes treatment`
   - Switch methods between BM25, TF-IDF, Dense, and Hybrid
   - Toggle between ranked view and grouped-by-topic view

2. Compare page
   - Show the same query across all four methods
   - Point out overlapping PMIDs and different evidence neighborhoods

3. Explore page
   - Show topic clusters for the curated PubMed subset
   - Explain that this is the data mining side of the project, not only ranking

4. Evaluation page
   - Show the unified dataset metrics
   - Mention exact PMID joining and BioASQ expert relevance labels

5. About page
   - Summarize the dataset story and practical local experimentation setup

## Suggested Demo Queries

- `heart attack`
- `myocardial infarction`
- `diabetes treatment`
- `drug adverse effects`
- `breast cancer biomarkers`

## Main Talking Points

- Documents come from a curated PubMed subset with PMIDs.
- Questions and relevance labels come from BioASQ Task B.
- The unified evaluation dataset uses exact PMID joins only.
- Search and clustering use the curated PubMed subset.
- Evaluation uses the unified PubMed-BioASQ dataset.
- BM25 is the strongest overall evaluation baseline in the current run.
- Hybrid is often the most balanced method.

## Optional AI-Assisted Layer

- Query rewrite suggestions
- Follow-up queries
- Match explanations
- Cluster label refinement

Important:
- This layer is clearly marked AI-assisted.
- It does not affect evaluation metrics.
- The app still works without it.
