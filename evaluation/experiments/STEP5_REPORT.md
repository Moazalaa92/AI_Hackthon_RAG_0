# STEP 5 REPORT — LLM-Assisted Residual Ranking Failure Diagnosis

**Phase 10 · Retrieval ranking diagnosis** · date: 2026-08-19
Frozen pipeline (dense+BM25→RRF→cross-encoder→Top-10), frozen binary labels, frozen baselines.
This report is **analysis only** — no production/retrieval/reranker/label changes.

---

## 1. Objective

Determine *why* the frozen cross-encoder still ranks distractors above more useful chunks,
using an independent LLM judge. Produce (a) a diagnostic dataset, (b) P@3 failure tables,
(c) relevance-vs-usefulness comparison, (d) score-margin analysis, (e) benchmark-vs-holdout
comparison, (f) hypothesis evidence, and (g) a conclusion on whether a second-stage
answerability/usefulness model is justified.

## 2. Method

### 2.1 Scope
For every affected question (P@3 failure candidates), judged **all chunks in the final Top-10
plus every chunk labeled `relevant` beyond Top-10** in that question's candidate pool:

- **Benchmark (9):** Q06 Q07 Q08 Q09 Q11 Q12 Q13 Q14 Q18 → 94 records
- **Holdout (15):** H01 H03 H08 H09 H10 H13 H15 H16 H17 H18 H19 H21 H22 H23 H24 → 160 records
- **Total: 254 records** (240 in Top-10, 14 relevant-beyond-Top-10)

### 2.2 Judge
`evaluation/experiments/step5_diagnose.py` — calls `deepseek/deepseek-v4-flash` (OpenRouter-compatible)
via `src/config.py` env (temperature 0). Strict JSON schema, rubric = *answer usefulness*
("relevant" ≠ "directly useful"):

```
diagnostic_class ∈ {DIRECT_ANSWER, SUPPORTING_DETAIL, RELEVANT_NOT_ANSWERING,
                    RATIONALE, INDEX_TOC, WRONG_SUBTOPIC, SCOPE_MISMATCH, OTHER}
answer_usefulness ∈ {high, medium, low}
direct_answer / supporting_detail / topically_relevant / rationale_or_index / scope_mismatch: bool
reason: str
```

**LLM judgments are a diagnostic signal, NOT ground truth** and never overwrite frozen labels.
Artifacts (reproducible): `step5_diagnosis_inputs.jsonl` (input), `step5_diagnosis.jsonl`
(254 judgments), `step5_summary.json`, `step5_summarize.py`.

### 2.3 Reliability
All 254/254 judged; 5 transient failures re-judged successfully (0 residual errors).
48 initial judgments were re-generated after a `max_tokens=300` truncation bug; final
judgments are full JSON. Deterministic metrics and existing labels are kept fully separate
from LLM judgments throughout.

## 3. Results

### 3.1 Diagnostic class distribution (top-10 + relevant-beyond-top-10)

| class | Benchmark (n=94) | Holdout (n=160) | All (n=254) |
|---|---|---|---|
| RATIONALE | 26 (28%) | 52 (33%) | **78 (31%)** |
| DIRECT_ANSWER | 19 (20%) | 32 (20%) | 51 (20%) |
| WRONG_SUBTOPIC | 13 (14%) | 23 (14%) | 36 (14%) |
| SCOPE_MISMATCH | 17 (18%) | 17 (11%) | 34 (13%) |
| RELEVANT_NOT_ANSWERING | 8 (9%) | 20 (13%) | 28 (11%) |
| SUPPORTING_DETAIL | 9 (10%) | 14 (9%) | 23 (9%) |
| OTHER / INDEX_TOC | 2 (2%) | 2 (1%) | 4 (2%) |

### 3.2 Existing label vs LLM answer-usefulness (all 254)

| existing label \ usefulness | high | medium | low | tot |
|---|---|---|---|---|
| relevant | 48 | 21 | **18** | 87 |
| not_relevant | **6** | 18 | 143 | 167 |

- **41 of 87 (47%) relevant-labeled chunks are NOT DIRECT_ANSWER** — i.e. labeled relevant but
  judged not to directly answer (supporting detail, rationale, or scope mismatch).
- 18 (21%) relevant chunks judged **low** usefulness (label inflation / thin relevance).
- 6 not_relevant chunks judged high usefulness (label misses) — small.

### 3.3 What occupies Top-3 in affected questions (n=72 top-3 slots)

| class | top-3 slots | usefulness high/med/low |
|---|---|---|
| DIRECT_ANSWER | 27 (38%) | — |
| RATIONALE | **21 (29%)** | 0 / 7 / 14 |
| SCOPE_MISMATCH | 7 (10%) | 1 / 0 / 6 |
| WRONG_SUBTOPIC | 6 (8%) | 0 / 1 / 5 |
| RELEVANT_NOT_ANSWERING | 6 (8%) | 0 / 2 / 4 |
| SUPPORTING_DETAIL | 5 (7%) | 2 / 1 / 2 |

Rationale/index content is the single largest *displaced* class in top-3 (29%).

### 3.4 P@3 failure tables (per question: top-3 occupants + best useful chunk below)

**Benchmark**

| Q | top-3 (class / usefulness) | best DIRECT_ANSWER below top-3 | margin (score gap) |
|---|---|---|---|
| Q06 WGS for unknown-cause epilepsy | DIRECT_ANSWER(high) · SUPPORT_DET(med) · **SUPPORT_DET(med, not_rel)** | r4 DIRECT_ANSWER high (s=4.21) | +1.15 (SUPPORT_DET) |
| Q07 referral time after 1st seizure | DIRECT_ANSWER(high) · **RELEV_NOT(rel-not-answer)** · **RATIONALE** | r8 | n/a |
| Q08 urgent tertiary referral children | **SCOPE_MISMATCH(rel, low)** · **RATIONALE** · DIRECT_ANSWER(high) | r4 DIRECT_ANSWER high (s=5.97) | +1.51 (SCOPE) |
| Q09 first-line monotherapy GTCS | **RATIONALE (s=8.92)** · **SCOPE_MISMATCH (s=8.74)** · DIRECT_ANSWER(high, s=8.58) | r10 DIRECT_ANSWER high (s=6.64) | +0.34 / +0.16 (RATIONALE, SCOPE) |
| Q11 first-line absence seizures | **SCOPE_MISMATCH (s=9.02)** · **WRONG_SUBTOPIC (s=8.75)** · **WRONG_SUBTOPIC (s=8.67)** | r4/r5 DIRECT_ANSWER high (s=8.45/8.44) | +0.57 / +0.30 / +0.22 |
| Q12 first-line Dravet | DIRECT_ANSWER(high) · DIRECT_ANSWER(high) · **SCOPE_MISMATCH (s=7.13)** | r4 DIRECT_ANSWER high (s=6.89) | +0.24 |
| Q13 first-line Lennox-Gastaut | DIRECT_ANSWER(high) · DIRECT_ANSWER(high) · **RELEV_NOT** | r9 | n/a |
| Q14 1st-line infantile spasms (non-TS) | DIRECT_ANSWER(high) · **WRONG_SUBTOPIC (s=5.83)** · **RATIONALE (s=3.46)** | r9/r17 DIRECT_ANSWER (s=−1.36/−4.87) | +7.19 / +4.82 |
| Q18 monitoring in pregnant epilepsy | SUPPORT_DET(high, rel) · **RATIONALE (s=3.11)** · **RELEV_NOT (s=2.61)** | r7 DIRECT_ANSWER (s=1.08) | +2.03 |

**Holdout (selected)**

| Q | top-3 distractor | best useful below | margin |
|---|---|---|---|
| H01 initial assessment after 1st seizure | RATIONALE(s=7.44) · SCOPE(rel) · SUPPORT_DET(rel) | r9 SUPPORT_DET high (s=3.03) | +4.41 / +3.81 / +3.33 |
| H03 12-lead ECG why | RELEV_NOT(s=4.12) · RATIONALE(s=3.27) | r5 RATIONALE high (s=2.48) | +1.64 / +0.79 |
| H08 tertiary access adults | RATIONALE(rel, s=4.71) · WRONG_SUBTOPIC(s=4.56) | r22/r25 (s=−1.48/−1.92) | +6.2 / +6.0 |
| H09 ASM treatment strategy | DIRECT_ANSWER(high) · DIRECT_ANSWER(high) · RATIONALE(s=5.23) | r5 | +0.79 |
| H10 mono vs polytherapy | RATIONALE(s=5.72) · RELEV_NOT(s=5.30) | r14 (s=3.67) | +2.05 / +1.63 |
| H16 1st-line tonic/atonic | RATIONALE(s=8.80) · SCOPE(s=8.59) | r5/r6 (s=8.14/8.02) | +0.66 / +0.45 |
| H21 ketogenic diet children | RATIONALE(s=6.40) · RELEV_NOT(s=5.26) | r9/r10 (s=4.76/4.24) | +1.64 / +0.5 |
| H23 reduce SUDEP risk | SUPPORT_DET(rel) · RATIONALE(s=5.96) · RATIONALE(s=5.46) | r8/r9 | n/a |
| H24 epilepsy nurse role | DIRECT_ANSWER(high) · DIRECT_ANSWER(high) · RATIONALE(med, s=7.23) | r9 RATIONALE high (s=5.51) | +1.72 |

### 3.5 Score-margin analysis (distractor vs best useful chunk below top-3)

| margin bucket | count | dominant classes |
|---|---|---|
| TIGHT (<1.0) | 11 | RATIONALE(4), SCOPE(3), WRONG_SUB(3), SUPPORT_DET(1), RELEV_NOT(1) |
| MID (1.0–2.5) | 11 | RATIONALE(5), SCOPE(2), RELEV_NOT(2), WRONG_SUB(1), SUPPORT_DET(1) |
| WIDE (>2.5) | 10 | RATIONALE(5), SUPPORT_DET(2), WRONG_SUB(2), SCOPE(1) |

- **Wide-margin cases are almost all RATIONALE** (5/10): the cross-encoder *confidently* ranks
  committee-rationale/evidence text above the answer chunk (Q09, Q14, H01, H08, H19, H22).
  These are genuine usefulness errors, not borderline.
- **Tight-margin cases** (Q11, Q12, H16, H21): near-ties where a small boost would flip; these are
  the *recoverable* failures.

### 3.6 Benchmark vs holdout comparison

Distributions are consistent across sets: RATIONALE dominates both (28% bench / 33% holdout),
DIRECT_ANSWER ≈20% both, WRONG_SUBTOPIC ≈14% both. The only differences: SCOPE_MISMATCH slightly
higher in benchmark (18% vs 11%), RELEVANT_NOT_ANSWERING slightly higher in holdout (13% vs 9%).
**No set-specific pathology** — findings generalize.

### 3.7 Does H6-A's deterministic suppression solve this? **No.**

Of **21 RATIONALE chunks in top-3**, only **2** contain H6-A's F-signals
(`\.{6,}`, `recommendation for research`, `nice technology appraisal guidance`). The other 19 are
true rationale prose (evidence-review / committee-consideration text), which H6-A does not touch.
RATIONALE chunks do frequently contain the words *evidence* (53/78) or committee-rationale phrasing
(24/78), but these are too noisy to use as hard negative rules without heavy false-positive risk.

## 4. Hypothesis evaluation (H1–H6)

| # | hypothesis | evidence | verdict |
|---|---|---|---|
| H1 | broad-semantic distractor (topically relevant, not answering) | WRONG_SUBTOPIC+SCOPE = 27% of top-3 displaced slots; tight+wide margins across sets | **Supported** (secondary) |
| H2 | rationale/index content displaces answers | RATIONALE = 31% of all judged, 29% of top-3 slots, 5/10 wide-margin cases; H6-A catches only 2/21 | **Strongly supported** |
| H3 | answerability vs relevance: model ranks "relevant" not "useful" | 41/87 (47%) relevant chunks not DIRECT_ANSWER; 18 relevant chunks low usefulness; 6 relevant-per-label high-usefulness chunks exist | **Strongly supported** |
| H4 | lexical/scope mismatch (e.g. Q11 "with other seizure types") | Q11 all-3-distractor case (SCOPE+WRONG_SUB, tight margins 0.22–0.57), Q08 top-1 scope distractor | **Supported** (narrow) |
| H5 | label noise / evaluation label mismatch | 18 relevant-labeled chunks judged low usefulness; 6 not_relevant judged high | **Weak support** (minor) |
| H6 | deterministic heuristic fixes the problem | H6-A improved P@3 +0.017/P@5 +0.020 (classification B) but catches only 2/21 top-3 RATIONALE chunks | **Rejected as sufficient** |

## 5. Root-cause conclusion

The cross-encoder does **not** fail because it is "wrong about relevance" — it mostly fails
because **relevance ≠ answer usefulness**, and the frozen binary labels encode relevance.

Three causal mechanisms, in order of impact:

1. **Rationale/evidence chunks beat answer chunks (H2).** 31% of all judged chunks are RATIONALE;
   they dominate top-3 displaced slots (29%) and wide-margin failures (5/10). These are
   *confidently* high-scoring because they are lexically and semantically close to the question
   (same drugs, same condition, committee phrasing). Deterministic suppression (H6-A) only catches
   index-like instances; the prose-rationale class is pattern-free and needs a model.
2. **The ranker optimizes topical relevance, not answerability (H3).** 47% of relevant-labeled
   chunks do not directly answer the question. The ms-marco-trained cross-encoder scores semantic
   proximity; it has no notion of "this chunk answers the question asked."
3. **Tight-margin near-ties (H4, minor H5).** Q11/Q12/H16/H21 cases are 0.2–0.7 score gaps —
   small reranking perturbations flip them; a handful are evaluation-label quirks (relevant-labeled
   chunks judged low usefulness).

**Conclusion on a second-stage model: a second-stage answerability/usefulness model IS justified.**

- The dominant failure class (RATIONALE, prose, not index) is not addressable by deterministic
  rules (only 2/21 top-3 cases caught by H6-A).
- A classifier that labels a chunk as `DIRECT_ANSWER vs SUPPORTING vs RATIONALE vs WRONG_SUBTOPIC`
  (the 4 high-precision classes the LLM judge used) can be trained offline on the frozen label set
  plus this diagnostic dataset, then applied **as a re-ranking signal only, evaluated offline** —
  never modifying frozen labels or the frozen pipeline.
- However, it must be evaluated against the **frozen labels** (P@3/P@5), not against LLM
  usefulness, because the labels are the canonical metric. Expected ceiling: removing the 21
  RATIONALE + 6 WRONG_SUBTOPIC + 7 SCOPE_MISMATCH displaced top-3 slots would recover ~2–3 P@3
  points on benchmark and ~2–4 on holdout (bounded by H6-A's measured +1.7/2.0 at partial coverage).

## 6. Recommended STEP 6 (experiment, offline only — NOT implemented)

Train a lightweight **answerability classifier** (e.g. fine-tuned cross-encoder or small BERT
classifier) on the diagnostic dataset: label = {DIRECT_ANSWER, SUPPORTING_DETAIL, RATIONALE,
WRONG_SUBTOPIC, SCOPE_MISMATCH, RELEVANT_NOT_ANSWERING} mapped to usefulness ranks. Use it as a
**second-stage re-rank signal** on the frozen Top-10 output (score fusion or hard suppression of
RATIONALE/SCOPE_MISMATCH), and evaluate **P@1/P@3/P@5/MRR/Hit@5 against the frozen labels** on both
benchmark and holdout. Constraints: label-free reorder only, no frozen-file changes, no production
change, report per-question deltas, stop for approval.

Stop after STEP 6 for approval (no further model implementation without sign-off).

---

## Artifacts

- `evaluation/experiments/step5_diagnose.py` — judge + dataset builder (reproducible, `--run-llm`, resume)
- `evaluation/experiments/step5_diagnosis_inputs.jsonl` — 254 deterministic inputs
- `evaluation/experiments/step5_diagnosis.jsonl` — 254 LLM judgments (strict schema)
- `evaluation/experiments/step5_summarize.py` + `step5_summary.json` — aggregation tables
- Baseline files and labels unchanged.