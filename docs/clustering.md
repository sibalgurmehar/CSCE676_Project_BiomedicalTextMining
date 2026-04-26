# Stage 8 Clustering

This stage groups top retrieved PubMed documents to show the structure of information surfaced by each retrieval method.

## Features

- MiniBatchKMeans clustering over top retrieved documents
- Supports clustering in:
  - TF-IDF vector space
  - dense embedding space
- Produces for each cluster:
  - cluster id
  - representative keywords
  - representative documents
  - cluster size
- Computes:
  - silhouette score
  - cluster size distribution
  - representative term summaries
- Supports:
  - grouping search results by cluster
  - cluster-aware diversification by round-robin interleaving clusters

## Script

```bash
backend/.venv/bin/python scripts/cluster_retrieval_results.py \
  --query "diabetes treatment" \
  --method bm25 \
  --dataset-name unified \
  --top-k 30 \
  --num-clusters 5 \
  --vector-space tfidf
```

## Output

Default output path:

`eval/results/clustering/<query>_<method>_<vector_space>.json`

