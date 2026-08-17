# Experiment Notes

## Baseline retrieval configuration (frozen reference — Phase 13.1)

```text
Baseline (Phase 12):
  chunk_size = 500, chunk_overlap = 50
  embedding model = sentence-transformers/all-MiniLM-L6-v2 (unchanged across experiments)
  vector store = Chroma, collection "documents", L2 distance space
  default K = 5 (from .env TOP_K); evaluation uses K = 10
  retrieval method = similarity_search_with_score
```

Baseline metrics (Phase 12, all 20 questions labeled by LLM-as-a-Judge, deepseek-v4-flash
temp 0 — proxy for relevance, not clinical ground truth):

```text
Mean Precision@3 = 0.3833
Mean Precision@5 = 0.3100
```

Source artifacts: `evaluation/results/baseline_500_50_top10.jsonl` (frozen retrieval,
200 records), `evaluation/results/baseline_500_50_top10_labeled.jsonl` (200 labels),
`evaluation/metrics/baseline_500_50_p3.json`, `_p5.json`, `_topk.json`,
`evaluation/failures.md`. The baseline Chroma store `data/chroma_db/` must remain untouched;
experiments use their own isolated stores under `data/chroma_db/experiments/<config>/`.

## Baseline 500/50 — Top-3 vs Top-5 vs Top-10 analysis (Phase 12.8)

Source: `evaluation/results/baseline_500_50_top10_labeled.jsonl` (all 200 records labeled,
LLM-as-a-Judge, deepseek-v4-flash temp 0 — proxy for relevance, not clinical ground truth).
Analysis artifact: `evaluation/metrics/baseline_500_50_topk.json`.

### What the growth 3 -> 5 -> 10 reveals

- **Rank 1 carries the load.** The first relevant chunk sits at rank 1 in 13/20 questions.
  16/20 questions have relevant material inside the top 3.
- **Ranks 4-5 add little on top of top-3.** Only 2 questions (Q11, Q18) find their *first*
  relevant chunk at rank 4; none at rank 5. The top-5 keeps 18/20 questions covered but the
  extra material is mostly noise.
- **Ranks 6-10 add noise, one exception.** Going from top-5 (18 questions covered) to top-10
  (19 questions covered) adds only one *first-seen* relevant chunk (Q07, rank 9). The other
  ranks 6-10 are almost all `not_relevant`.
- **Precision falls as K grows** (mean P@3 0.383 -> P@5 0.310 -> P@10 0.211): the added ranks
  are increasingly irrelevant — the classic noise frontier.

### Evidence for Phase 12.9 failure analysis

Relevant chunk missing from top-10: 1 question (Q20 — deliberate unanswerable dosing case,
expected zero relevant, so it is NOT a retriever failure).

Relevant chunk first seen below rank 3 (candidate genuine failures to investigate in 12.9):

- Q07: first relevant at rank 9.
- Q11: first relevant at rank 4 (relevant also at rank 7).
- Q18: first relevant at rank 4.
- Q17: first relevant at rank 2 (only one relevant in top-10).

These are candidates only; Phase 12.9 will verify each case before labeling it a genuine
retrieval failure. No chunk-size conclusions are drawn from this phase alone.
