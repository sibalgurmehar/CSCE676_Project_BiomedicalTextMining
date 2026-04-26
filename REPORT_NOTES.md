# Report Notes

## Final Report Story

The report can follow this order:

1. BioASQ questions and relevant PubMed IDs
2. Curated PubMed subset with PMIDs
3. Exact PMID-based joining
4. Retrieval evaluation
5. Query-type analysis
6. Clustering and diversity analysis

## Suggested Section Outline

### 1. Problem

BioSeek investigates how retrieval method choice changes not only relevance, but also the breadth and structure of biomedical evidence retrieved from PubMed.

### 2. Data Sources

- BioASQ Task B for biomedical questions and expert relevance judgments
- Curated PubMed subset for the document corpus

### 3. Dataset Construction

- Load BioASQ questions and relevant PMIDs
- Build a manageable PubMed subset for local experimentation
- Preserve PMIDs exactly
- Perform exact PMID-based joining
- Keep only surviving queries with at least one mapped relevant document

### 4. Retrieval Methods

- TF-IDF
- BM25
- Dense semantic retrieval
- Hybrid retrieval

### 5. Evaluation

Use the unified PubMed-BioASQ dataset and report:

- Precision@5
- Precision@10
- Recall@10
- MRR
- nDCG@5
- nDCG@10

### 6. Query-Type Analysis

Use interpretable buckets:

- exact terminology
- synonym-heavy
- treatment / intervention
- outcome / effect
- broad exploratory

### 7. Clustering and Diversity

- cluster top retrieved PubMed documents
- inspect topic structure by method
- compare cluster spread and theme coverage

### 8. Findings

Use the current result story:

- BM25 performs best overall on the unified benchmark
- lexical retrieval is strongest when terminology overlap is high
- hybrid is more robust than dense alone
- diversity analysis shows meaningful differences in how broad each method’s retrieved evidence is

### 9. Limitations

- curated subset rather than the full PubMed collection
- student-scale local experimentation constraints
- dense adaptation and AI-assisted features are optional layers, not core benchmark components

## One-Sentence Thesis

Retrieval method choice changes not only which PubMed articles rank highest, but also how broad, diverse, and structurally informative the retrieved biomedical evidence becomes.
