# Optional Dense Adaptation

Stage 7 is intentionally optional. The goal is to try a small biomedical dense-retriever adaptation without turning the project into a full model-training project.

## Design

- Base model: `sentence-transformers/all-MiniLM-L6-v2`
- Positive pairs:
  - PubMed `title` to `abstract`
  - lightweight near-topic pairs from documents sharing a first MeSH term
- Negative pairs:
  - random unrelated PubMed document pairs
- Training:
  - small PyTorch loop with `CosineEmbeddingLoss`
  - one or a few epochs only

## Files

- Config: `eval/configs/dense_adaptation_config.json`
- Training script: `scripts/train_adapted_dense.py`
- Comparison script: `scripts/compare_adapted_dense.py`
- Checkpoints: `eval/checkpoints/dense_adaptation/`

## Suggested workflow

```bash
cd backend
source .venv/bin/activate
pip install -r requirements-stage7.txt
cd ..
backend/.venv/bin/python scripts/train_adapted_dense.py
backend/.venv/bin/python scripts/compare_adapted_dense.py
```

## Reporting rule

If the adapted model does not outperform the base dense retriever, report that result honestly. In this project, the adaptation is an optional experiment, not a required success condition.

