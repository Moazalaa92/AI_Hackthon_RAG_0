# Experiment Notes

## How to read this file

Each experiment section records: **configuration**, **reason for experiment**, **result**,
and **decision**. Metrics use LLM-as-a-Judge labels (deepseek-v4-flash, temp 0 — a proxy for
relevance, not clinical ground truth). Detailed failure analyses live in
`evaluation/metrics/comparison.md` and `evaluation/failures.md`; this file is the
hypothesis → decision log.

## Evaluation conventions (all experiments)

- Corpus: `data/pdfs/Project_pdf.pdf` (NICE NG217, 161 pages).
- Benchmark dataset: `evaluation/dataset.json` (Q01–Q20; Q20 is a deliberate negative).
- Holdout dataset: `evaluation/holdout_dataset_v1.json` (H01–H27; H26/H27 deliberate
  negatives), created after the experiments to test generalization.
- Vector store: Chroma, collection `documents`, **L2 distance space** (lower = more similar
  for dense; `score` in hybrid = RRF fusion, higher = better; `score` in reranked = raw
  cross-encoder logit, higher = better).
- Labels: LLM judge in `src/judge.py`, rubric in the module, written to derivative
  `<name>_labeled.jsonl` files (frozen retrieval JSONL never modified).
- Metrics: **Precision@K** (relevant in top-K / K), **Hit@K** (≥1 relevant in top-K),
  **candidate recall** (relevant evidence in the union candidate pool). None of these is
  "answer accuracy".

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
Hit@3 = 0.80 | Hit@5 = 0.90 | Hit@10 = 0.95
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

## Experiment history (Phase 13 — the optimization lab)

All experiments are measured against the Phase 12 baseline with the SAME 20 benchmark
questions, same judge, same top-10. One variable changes per experiment (golden rule).

### Experiment 1 — Chunking: 800/100 vs 500/75 (13.2/13.3)

- **Configuration:** `large_800_100` (444 chunks) and `large_500_75` (653 chunks), MiniLM
  embeddings, isolated Chroma stores under `data/chroma_db/experiments/`.
- **Reason:** baseline failures (Q07 answer truncated at the 500-char boundary, Q11/Q18
  answers at chunk ends) suggested larger chunks / more overlap would keep each
  recommendation in one chunk.
- **Result:** 800/100 P@3 **0.4167** (Δ +0.033), P@5 0.31, Hit@3 0.90; 500/75 P@3 0.4167,
  P@5 0.32, Hit@3 0.85. 800/100 fully fixed Q11 and improved Q18; 500/75 fixed Q07 but
  regressed Q18. Neither changed Hit@5/Hit@10 (0.90/0.95) or the only no-relevant question
  (Q20, the deliberate negative).
- **Decision:** **800/100 is the selected chunk configuration.** Chunking alone did not solve
  the remaining ranking problems (residual failures were paraphrase / lexical-distractor
  cases).

### Experiment 2 — Embedding: all-mpnet-base-v2 on 800/100 (13.5)

- **Configuration:** same 800/100 chunks, Chroma, retrieval API, dataset, judge; only the
  embedding changed (MiniLM → mpnet, 768-dim).
- **Reason:** the residual failure class (semantic paraphrase + lexically similar
  distractors, e.g. Q18 "closer supervision" vs "more frequent monitoring") was hypothesized
  to be an embedding-capacity problem.
- **Result:** P@3 0.3833, P@5 0.28, Hit@3 0.85, Hit@5 0.85, Hit@10 0.90 — **flat-to-worse**
  than 800/100 MiniLM (0.4167). mpnet fixed the single worst case (Q07 9→1) but regressed
  Q16 and left Q09's true answer out of Top-10.
- **Decision:** **keep MiniLM.** The larger model does not improve aggregate quality and adds
  ~4× inference cost.

### Experiment 3 — Hybrid retrieval: dense + BM25 + RRF on 800/100 (13.5)

- **Configuration:** dense top-20 (Chroma/L2) + BM25 top-20 (rank_bm25 over the same 444
  stored chunks) → union → RRF (k=60) → top-10.
- **Reason:** several failures were lexical-overlap / terminology cases (Q09 MRI question
  returning treatment chunks; Q16 topiramate vs valproate) that a keyword signal directly
  attacks, and Q09's answer was absent from the dense-only candidate pool.
- **Result:** P@3 0.4167 (unchanged), P@5 **0.35** (Δ +0.04). **Candidate recall improved:**
  relevant in Dense Top-20 = 18/20, BM25 Top-20 = 19/20, union = 19/20; 9 of 57 relevant
  chunks (≈16%) were BM25-only, including Q09 — the only configuration to find any Q09
  relevant chunk. Q09's BM25-only chunks fused to ranks 13/27 (RRF cap) — recall solved,
  ranking not. Q18 regressed (fusion pushed its relevant chunk down).
- **Decision:** hybrid expands candidate recall but RRF's rank-based cap hurts BM25-only
  chunks. **Next: a learned reranker over the union pool.**

### Experiment 4 — Reranking: cross-encoder over the hybrid pool (13.5)

- **Configuration:** frozen hybrid union pool (615 candidates) re-scored by
  `cross-encoder/ms-marco-MiniLM-L-6-v2` (raw logits, higher = more relevant; the SOLE final
  ranking signal, RRF removed). No retrieval, no chunking, no embedding change.
- **Reason:** the remaining failures were rank-order problems within a pool that already
  contained the answers (Q09 at 13/27, Q18 at 4/6, Q16 at 2/3/5). A cross-encoder makes a
  fine-grained (question, chunk) relevance judgment that RRF cannot.
- **Result:** P@3 **0.5167** (Δ +0.10 vs hybrid — the largest single gain), P@5 0.40,
  Hit@3 0.90, Hit@5 0.95, Hit@10 0.95. Q09's BM25-only relevant chunk promoted to rank 3;
  Q18 restored to rank 1; Q16 improved. One regression: Q11 (cross-encoder over-ranked the
  "absence seizures **with other seizure types**" distractor to rank 1 — a lexical-overlap /
  domain-nuance failure, isolated to Q11). Judge-variance check: gain well above the ~5–7%
  label-noise floor.
- **Decision:** **this is the leading architecture: 800/100 + MiniLM + Dense Top-20 +
  BM25 Top-20 + cross-encoder → Top-10.** Proceed to validation on a held-out question set.

### Experiment 5 — Holdout validation (13.8)

- **Configuration:** the frozen leading architecture run against a NEW 27-question holdout
  (`evaluation/holdout_dataset_v1.json`; 25 answerable + 2 deliberate negatives), created
  independently of retrieval outputs and covering guideline sections absent from the
  benchmark.
- **Reason:** the 20-question benchmark is small and the Q11 failure might be a one-off; a
  holdout tests whether the architecture generalizes.
- **Result:** P@3 **0.5802**, P@5 0.4519, Hit@3/5/10 = 0.9259 (25/27; the two misses are the
  deliberate negatives). Answerable-only Hit@3/5/10 = 25/25 = 1.0. Candidate recall 25/25
  answerable = 100%. First-relevant at rank 1 for 22/25, rank 2 for 3/25 — **no Q11-style
  regression occurred**. Holdout P@3 Wilson 95% CI [0.472, 0.682] vs benchmark [0.393,
  0.638]: at or above the benchmark on every metric, never below.
- **Decision:** the architecture **generalizes**. Candidate recall is strong enough to move
  forward. **Retrieval is VALIDATED / FROZEN FOR NOW.** Remaining issue is Top-3 precision
  density (more relevant chunks per top-3), which is optimization, not a blocker.

## Current retrieval diagnosis (summary)

```text
Original problem:      not simply missing answers
Chunking:              partially improved context fragmentation (800/100 selected)
Embedding:             larger model did not improve aggregate quality (MiniLM kept)
BM25:                  expanded candidate recall (union 19/20, 25/25)
Cross-encoder:         substantially improved ranking (P@3 0.4167 -> 0.5167)
Remaining issue:       Top-3 precision density / occasional ranking/context confusion
Candidate recall:      strong enough to move forward (19/19 benchmark, 25/25 holdout)
```

## Next step

**Phase 8 — Generation** (NOT STARTED). Do NOT change the frozen retrieval architecture
(chunks, embedding, dense/BM25 depths, cross-encoder, top-10 depth) before Generation.
(RRF was used only in the historical hybrid experiment; it is NOT part of the frozen
architecture and must not be reintroduced.)
