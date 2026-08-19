# Generation Baseline Evaluation (Phase 8.5)

Measurement of the Phase 8 generation baseline as it exists — **no optimization,
no prompt/model/retrieval changes** were made in this phase.

## Methodology

- **Pipeline measured:** question → frozen retrieval (dense Top-20 + BM25 Top-20 →
  cross-encoder → Top-10) → context construction → grounding prompt →
  `deepseek/deepseek-v4-flash` (temperature 0) → answer.
- **Retrieval:** the FROZEN canonical outputs are read-only inputs. Benchmark
  questions use `evaluation/results/reranked_hybrid_800_100_top10_labeled.jsonl`
  (Q01–Q20) and holdout questions use
  `evaluation/results/holdout_v1_reranked_hybrid_top10_labeled.jsonl` (H01–H27).
  No retrieval artifact was modified; relevance labels come from the frozen
  Phase 12.5/13 LLM judge.
- **Dataset:** `evaluation/generation_dataset_v1.json` — all 47 questions (44
  answerable, 3 deliberate negatives), with curated reference answers derived
  from the frozen datasets' `notes` and the guideline content.
- **Generated answers:** `evaluation/results/generation_v1.jsonl` (47 records).
- **Judged results:** `evaluation/results/generation_v1_judged.jsonl` (47 records).

## LLM-as-a-judge (proxy, not ground truth)

- Judge model: `deepseek/deepseek-v4-flash`, temperature 0, `max_tokens=2000`,
  3 retries on empty/malformed responses (`src/generation_judge.py`).
- Four SEPARATE judge calls (no leakage across dimensions):

| Dimension | Judge receives | Judge does NOT receive |
|---|---|---|
| Correctness | question, reference answer, generated answer | retrieved context |
| Grounding | question, retrieved context, generated answer | reference answer |
| Completeness | question, reference answer, retrieved context, answer | — |
| Abstention | question, retrieved context, generated answer | reference answer |

- **Limitations (same principle as the Phase 12.5 retrieval judge):** an LLM
  judge is a proxy, not proof of correctness. Labels depend on the reference
  answers (for correctness/completeness) and on the judge's own interpretation.
  This establishes an engineering baseline, not a clinical certification.

## Dataset

| Group | Answerable | Deliberate negatives | Total |
|---|---|---:|---:|---:|
| Benchmark (Q01–Q19 + Q20) | 19 | 1 | 20 |
| Holdout (H01–H25 + H26, H27) | 25 | 2 | 27 |
| **Total** | **44** | **3** | **47** |

## Results (n = 44 answerable + 3 negatives)

| Metric | Result |
|---|---:|
| **Factual correctness — correct** | **30 / 44 (68.2%)** |
| Partially correct | 11 / 44 (25.0%) |
| Incorrect | 3 / 44 (6.8%) |
| **Grounding — fully grounded** | **44 / 44 (100%)** |
| Partially grounded | 0 / 44 |
| Unsupported | 0 / 44 |
| **Completeness — complete** | **39 / 44 (88.6%)** |
| Partial | 4 / 44 (9.1%) |
| Incomplete | 1 / 44 (2.3%) |
| **Abstention — appropriate refusal** | **3 / 3 (100%)** |
| Inappropriate answer (negative cases) | 0 / 3 |

Breakdown by subset (correctness):

| Subset | Correct | Partially correct | Incorrect |
|---|---|---:|---:|---:|
| Benchmark (n=19) | 15 (78.9%) | 4 (21.1%) | 0 (0%) |
| Holdout (n=25) | 15 (60.0%) | 7 (28.0%) | 3 (12.0%) |

## Failure attribution (incorrect + partially-correct answerable cases)

| Failure type | Count |
|---|---:|
| No failure (correct) | 30 |
| Retrieval failure (no relevant chunk in Top-10) | 0 |
| Generation failure (relevant evidence present, answer wrong/partial) | 14 |

Every answerable question had at least one relevant chunk in the frozen Top-10
(`evidence_present = true` for all 44). **Zero failures are attributable to
retrieval** at the generation stage in this set; all 14 non-correct answers are
generation-stage failures.

See `evaluation/generation_failures.md` for the per-question failure analysis.
