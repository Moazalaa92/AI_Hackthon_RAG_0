# STEP 6 REPORT — Second-Stage Answerability/Usefulness Reorder Experiment

**Phase 10 · Retrieval ranking diagnosis** · date: 2026-08-19
**OFFLINE EXPERIMENT ONLY** — the canonical pipeline, frozen artifacts, labels, baselines,
datasets, and production code are all **unchanged**.

---

## 1. Objective

Test the STEP 5 hypothesis experimentally:

> "The residual ranking problem is partly caused by the distinction between retrieval relevance
> and answer usefulness. A second-stage answerability/usefulness signal may improve the ordering
> of the existing Top-10 chunks."

STEP 6 must determine, by experiment, whether such a signal actually improves ranking quality —
measured **only** against the frozen binary relevance labels.

## 2. Frozen baseline (unchanged, reproduced exactly)

| set | P@1 | P@3 | P@5 | MRR | Hit@5 | n |
|---|---|---|---|---|---|---|
| Benchmark | 0.850 | 0.533 | 0.390 | 0.879 | 0.950 | 20 |
| Holdout | 0.815 | 0.580 | 0.452 | 0.870 | 0.926 | 27 |

Both reproduced to 4 decimals from the frozen candidate/label files
(`run_step6.py` baseline check).

## 3. Data used

- Frozen reranked candidates + frozen labels (bench 615, hold 858 records).
- STEP 5 LLM diagnostic judgments (254 records: 9 affected benchmark + 15 affected holdout
  questions; top-10 + relevant-beyond-top-10 chunks).
- All STEP 6 artifacts are **new**, under `evaluation/experiments/step6/`. No frozen file modified.

## 4. Leakage analysis

**Experiment scope:** reorder only the frozen Top-10 per question (a permutation of the same 10
chunks; nothing pulled in/out, candidate pool untouched). Non-affected questions keep frozen order.

**Oracle variants are leaky by construction.** `oracle_usefulness`/`oracle_class` use the LLM
judgment *of the very question being evaluated* as the answerability score. This makes them
**ORACLE / UPPER-BOUND experiments**: they answer *"how much headroom exists if usefulness were
perfectly known?"* and must NOT be read as achievable with a real model.

**Non-oracle variants avoid leakage.**
- `lqo` (leave-question-out): for each affected question, a classifier is trained on all diagnostic
  records **from other questions only** and predicts usefulness for the held-out question's chunks.
- `crossset`: trained on benchmark affected questions, applied to holdout (and vice versa).
- `suppress_rationale`: class-based hard demotion using the LLM diagnostic class of *the same
  question* → this is also leaky/oracle (flag noted in §6).

LLM usefulness is **never** used as the evaluation metric. All metrics are frozen-label P@1/3/5,
MRR, Hit@5.

## 5. Scoring formulation

Per-question combined score (investigated: per-question min-max normalization, since reranker
score spreads vary 1.5→12.9 across questions; the usefulness score is already on [0,1]):

```
norm_ce = minmax(reranker_score) within the question's top-10 pool
combined = (1 - alpha) * norm_ce + alpha * norm(answerability_score)
```

Answerability score sources:
- `oracle_usefulness`: high=1.0, medium=0.5, low=0.0.
- `oracle_class`: DIRECT_ANSWER=1.0, SUPPORTING_DETAIL=0.7, RELEVANT_NOT_ANSWERING=0.4,
  WRONG_SUBTOPIC=0.2, SCOPE_MISMATCH=0.2, RATIONALE=0.1, INDEX_TOC=0.0, OTHER=0.3.
- `lqo`/`crossset`: predicted usefulness in [0,1] from a LogisticRegression over
  `all-MiniLM-L6-v2` embeddings of (question, chunk) pairs, trained per §4.

Weight sweep: **alpha ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}** for oracle/lqo/crossset.

## 6. Experiments run

| variant | type | leakage |
|---|---|---|
| baseline | frozen | — |
| oracle_usefulness (α sweep) | ORACLE | uses target question's LLM usefulness |
| oracle_class (α sweep) | ORACLE | uses target question's LLM class |
| lqo (α sweep) | NON-ORACLE | leave-question-out classifier |
| crossset (α sweep) | NON-ORACLE | train bench→apply hold & train hold→apply bench |
| suppress_rationale | ORACLE (leaky class signal) | hard-demotes RATIONALE/INDEX_TOC |

## 7. Weight sweep — full-set results (delta vs baseline)

| variant | bench P@3 (Δ) | bench P@1 (Δ) | hold P@3 (Δ) | hold P@1 (Δ) |
|---|---|---|---|---|
| oracle_usefulness α=0.1 | .583 (+.050) | .850 (0) | .605 (+.025) | .815 (0) |
| oracle_usefulness α=0.2 | .617 (+.083) | .950 (+.100) | .630 (+.049) | .815 (0) |
| oracle_usefulness α=0.3 | .600 (+.067) | .950 (+.100) | .654 (+.074) | .852 (+.037) |
| **oracle_usefulness α=0.5** | **.650 (+.117)** | **.950 (+.100)** | **.679 (+.099)** | **.852 (+.037)** |
| oracle_usefulness α=0.7 | .650 (+.117) | .950 (+.100) | .691 (+.111) | .889 (+.074) |
| oracle_class α=0.5 | .633 (+.100) | .950 (+.100) | .691 (+.111) | .889 (+.074) |
| lqo α=0.2 | .550 (+.017) | .850 (0) | .580 (0) | .852 (+.037) |
| lqo α=0.3 | .567 (+.033) | .900 (+.050) | .605 (+.025) | .852 (+.037) |
| lqo α=0.5 | .567 (+.033) | .850 (0) | .617 (+.037) | .815 (0) |
| lqo α=0.7 | .517 (−.017) | .650 (−.200) | .605 (+.025) | .778 (−.037) |
| crossset α=0.3 | .567 (+.033) | .800 (−.050) | .605 (+.025) | .852 (+.037) |
| crossset α=0.5 | .567 (+.033) | .750 (−.100) | .617 (+.037) | .778 (−.037) |
| suppress_rationale | .550 (+.017) | .850 (0) | .605 (+.025) | .889 (+.074) |

## 8. Benchmark results (affected-subset, where reordering applies)

Baseline on affected 9 questions: P@1 .778, P@3 .444, P@5 .378, MRR .843.

| variant | P@1 (Δ) | P@3 (Δ) | P@5 (Δ) | MRR (Δ) |
|---|---|---|---|---|
| oracle_usefulness α=0.5 | 1.000 (+.222) | **.704 (+.259)** | .444 (+.067) | 1.000 (+.157) |
| oracle_class α=0.5 | 1.000 (+.222) | .667 (+.222) | .444 (+.067) | 1.000 (+.157) |
| lqo α=0.3 | .889 (+.111) | .519 (+.074) | .400 (+.022) | .917 (+.074) |
| lqo α=0.5 | .778 (0) | .519 (+.074) | .400 (+.022) | .861 (+.019) |
| crossset α=0.3 | .667 (−.111) | .519 (+.074) | .378 (0) | .806 (−.037) |
| suppress_rationale | .778 (0) | .481 (+.037) | .400 (+.022) | .861 (+.019) |

Per-question P@3 changes (oracle_usefulness α=0.5): improved **6/9** (Q06 .667→1.0, Q09 .333→.667,
Q11 0→.667, Q12 .667→1.0, Q14 .333→.667, Q18 .333→.667), unchanged 3/9 (Q07, Q08, Q13),
regressed **0/9**. P@1 regressions 0. **Q07/Q08/Q13 have too few relevant chunks in top-10 to gain
any P@3 even with a perfect oracle** (Q07 has 1 relevant in top-10; Q08's relevant are already 2
in top-3; Q13 similar). Oracle fixes **all recoverable benchmark failures**.

## 9. Holdout results (affected-subset)

Baseline on affected 15 questions: P@1 .867, P@3 .533, P@5 .440, MRR .933.

| variant | P@1 (Δ) | P@3 (Δ) | P@5 (Δ) | MRR (Δ) |
|---|---|---|---|---|
| oracle_usefulness α=0.5 | .933 (+.067) | **.711 (+.178)** | .533 (+.093) | .967 (+.033) |
| oracle_class α=0.5 | 1.000 (+.133) | .733 (+.200) | .547 (+.107) | 1.000 (+.067) |
| lqo α=0.3 | .933 (+.067) | .578 (+.044) | .467 (+.027) | .967 (+.033) |
| lqo α=0.5 | .867 (0) | .600 (+.067) | .453 (+.013) | .933 (0) |
| crossset α=0.3 | .933 (+.067) | .578 (+.044) | .440 (0) | .956 (+.022) |
| suppress_rationale | 1.000 (+.133) | .578 (+.044) | .520 (+.080) | 1.000 (+.067) |

Per-question P@3 (oracle_usefulness α=0.5): improved **7/15**, unchanged 8/15, regressed **0/15**,
P@1 regressions 0. oracle_class α=0.5 has 1 holdout regression (H08 .667→.333) → usefulness score
is safer than class mapping for holdout.

## 10. Per-question changes (oracle_usefulness α=0.5)

| Q | P@3 before→after | what changed |
|---|---|---|
| Q06 | .667→1.000 | DIRECT_ANSWER promoted to top-2 (was SUPPORTING_DETAIL) |
| Q09 | .333→.667 | DIRECT_ANSWER moved to rank 1 (RATIONALE 8.92→rank 2) |
| Q11 | 0→.667 | both DIRECT_ANSWERs into top-3 (SCOPE/WRONG_SUBTOPIC demoted) |
| Q12 | .667→1.000 | SCOPE_MISMATCH (7.13) demoted below 2 DIRECT_ANSWERs |
| Q14 | .333→.667 | WRONG_SUBTOPIC/RATIONALE demoted; DIRECT_ANSWERs up |
| Q18 | .333→.667 | RATIONALE demoted; DIRECT_ANSWER rank 7→top-3 |
| Q07/Q08/Q13 | unchanged | insufficient relevant-in-top-10 to move (Q07: 1 relevant total) |

Holdout similar: H16 .333→1.000, H18/H19/H22/H24 →1.000, H01/H08/H17/H23 fixed or improved at
class-level (see §6 note), H03/H09/H15/H21 improved. The one oracle_class regression (H08) is
caused by demoting a relevant RATIONALE chunk (H08 rank-2 chunk is labeled relevant AND judged
RATIONALE medium) — a genuine relevance/usefulness conflict, flagged in §14.

## 11. Regression analysis

- **Oracle variants (usefulness): 0 per-question regressions on benchmark and holdout at every α
  tested; 0 P@1 regressions.** This is the cleanest result: with perfect usefulness, gains are
  strictly non-negative in aggregate.
- **Non-oracle lqo:** at α=0.3, 1 benchmark regression (Q08 .667→.333) and 2 holdout regressions
  (H01, H17); 0 P@1 regressions. At α=0.5, 2-3 P@1 regressions begin to appear (Q07, H17, H19).
- **crossset:** 2 benchmark P@1 regressions (Q07, Q18) even at α=0.3 → cross-set transfer is the
  least stable.
- **suppress_rationale:** 2 holdout P@3 regressions (H08, H17) — relevant chunks that are
  RATIONALE get wrongly demoted. This confirms that **hard class suppression is not safe** and
  matches STEP 4's H6 finding (relevant chunks share rationale/scope phrasing).

## 12. Candidate recall

All variants reorder only the frozen Top-10, so the top-10 set (and hence the candidate pool
coverage) is **preserved exactly** — verified programmatically for every variant
(`top-10 set preserved=True`). Recall remains 100% (bench) and 100% (holdout).

## 13. Distractor-class analysis (which classes were actually fixed)

Top-3 class composition before → after (oracle_usefulness α=0.5), affected questions:

| class | bench before→after | hold before→after |
|---|---|---|
| DIRECT_ANSWER | 9 → **17** | 18 → **27** |
| RATIONALE | 5 → 3 | 16 → 9 |
| SCOPE_MISMATCH | 4 → 1 | 3 → 2 |
| WRONG_SUBTOPIC | 3 → 0 | 3 → 1 |
| RELEVANT_NOT_ANSWERING | 3 → 2 | 3 → 1 |
| SUPPORTING_DETAIL | 3 → 4 | 2 → 5 |

**The oracle gain comes from genuinely moving DIRECT_ANSWER chunks upward**, not from gaming label
quirks. RATIONALE is halved in holdout top-3; SCOPE_MISMATCH and WRONG_SUBTOPIC nearly eliminated
in benchmark. Relevant chunks demoted by the oracle: Q08/96 (labeled relevant, judged
SCOPE_MISMATCH low), H01/38 (relevant, SCOPE_MISMATCH low), H13/202 (relevant, RATIONALE low) —
3 relevant chunks moved out of top-3 across 24 affected questions, each replaced by a DIRECT_ANSWER
chunk that also contributes a frozen-label relevant hit (Q08's P@3 unchanged at .667, H01/H13
improved). Net: no frozen-label loss from these demotions.

## 14. Which relevant chunks were incorrectly demoted (non-oracle risk)

The main non-oracle regression source is **relevant chunks that are RATIONALE/SCOPE-mismatched
yet labeled relevant**. lqo α=0.3 demotes H08/194 (relevant RATIONALE) and Q08/96 (relevant
SCOPE_MISMATCH), causing Q08/H08 P@3 drops. This is the core tension: **the frozen labels call
these chunks relevant, but the usefulness view says they don't answer the question.** Any real
answerability model will repeatedly hit this 18-chunk (relevant-labeled but low-usefulness)
population; they are the price of the signal.

## 15. Comparison against H6-A (deterministic suppression, STEP 4)

| metric | baseline | H6-A | suppress_rationale | lqo α=0.3 | oracle_usefulness α=0.5 |
|---|---|---|---|---|---|
| bench P@3 | .533 | .550 (+.017) | .550 (+.017) | .567 (+.033) | .650 (+.117) |
| bench P@5 | .390 | .410 (+.020) | .400 (+.010) | .400 (+.010) | .420 (+.030) |
| hold P@3 | .580 | .580 (0) | .605 (+.025) | .605 (+.025) | .679 (+.099) |
| hold P@5 | .452 | .467 (+.015) | .496 (+.044) | .467 (+.015) | .504 (+.052) |

- H6-A (F1/F2/F3 string suppression) remains the only **safe deterministic** lever but is bounded:
  it catches only index-like rationale.
- The non-oracle classifier (lqo) at modest α matches or slightly beats H6-A on P@3/P@5 with
  comparable regression profile — but is far below the oracle.
- The oracle shows the *mechanism* (answerability weighting) is capable of **roughly doubling** the
  deterministic gain, **if** usefulness can be predicted well.

## 16. Does answerability scoring actually help? (answer to the STEP 5 hypothesis)

**ORACLE / UPPER-BOUND (PROVEN, but with perfect information):** yes. With per-question
min-max normalization and α=0.5, a usefulness-weighted reorder of the frozen top-10 delivers
benchmark P@3 **0.533→0.650** (+0.117), P@5 +0.030, P@1 +0.100; holdout P@3 **0.580→0.679**
(+0.099), P@5 +0.052. Zero per-question regressions. The gains are earned by promoting
DIRECT_ANSWER chunks and demoting RATIONALE/SCOPE_MISMATCH/WRONG_SUBTOPIC (top-3 DIRECT_ANSWER
composition rises 9→17 bench, 18→27 hold).

**EXPERIMENTALLY SUPPORTED, but only weakly, with a real (non-oracle) signal:** the
leave-question-out classifier — the honest test — yields only benchmark P@3 +0.033 and holdout
P@3 +0.025 at α=0.3, with 3-4 per-question regressions and (at higher α) P@1 regressions. The
classifier's agreement with the LLM usefulness oracle is weak (Pearson r≈0.26; median per-question
Spearman ≈0.28; negative correlation on 5/24 questions). **With 254 diagnostic examples the
signal is real but the model is too weak to exploit it reliably.**

**Conclusion:** the STEP 5 hypothesis is **EXPERIMENTALLY SUPPORTED in direction** (usefulness
weighting can improve ranking) but **NOT yet practically delivered** — no non-oracle variant
achieves the oracle's gains, and none clearly beats the safe deterministic H6-A once regressions
are priced in. Headroom exists; the bottleneck is signal quality, not the weighting scheme.

## 17. Is a trained second-stage model justified? (recommendation)

**Justified only conditionally — more diagnostic data is needed first.** Evidence:
- Oracle headroom is large and consistent across benchmark and holdout (PROVEN upper bound).
- The 254-sample LQO classifier is too weak (r≈0.26) to realize it; this is a data-volume
  problem, not a proof that the approach fails.
- Hard suppression of RATIONALE (suppress_rationale) is **not** safe (H08/H17 regressions) —
  a model-based signal, not a rule, is required, confirming STEP 5 §5.

Recommended path (NOT implemented, needs approval):
1. **Expand the diagnostic dataset** to all benchmark + holdout questions' top-10 (≈20+27 questions
   × 10 chunks ≈ 400+ records; + the current 254), reusing the frozen LLM judge, to give a
   leave-question-out classifier enough signal.
2. Re-run this exact STEP 6 harness on the enlarged dataset. Success criterion: the **non-oracle**
   lqo/crossset variants approach the oracle gains (P@3 +≥0.08 bench, +≥0.05 hold) with ≤1
   per-question regression per set and no P@1 regressions, on both sets.
3. Only then consider training a real second-stage answerability model (still as an offline
   reorder, never modifying the frozen pipeline).

**Do NOT promote any STEP 6 variant to production.** All results are experimental.

## 18. Recommended next step

`STEP 7 (proposed, pending approval)`: expand the LLM diagnostic dataset (benchmark + holdout
top-10, all 47 questions), retrain the LQO classifier on the larger set, and rerun this STEP 6
harness (identical scoring/normalization/α-sweep) to test whether the non-oracle signal closes the
gap to the oracle. If it does, evaluate a final candidate second-stage model offline. If it does
not, the answerability approach is **UNKNOWN/UNSUPPORTED** and we stop — no production change
either way.

---

## Evidence classification summary

| claim | status |
|---|---|
| Baseline metrics reproduce exactly | PROVEN |
| Oracle usefulness weighting improves P@3/P@1 on bench & hold, 0 regressions | PROVEN (ORACLE / UPPER-BOUND) |
| Top-3 gain comes from DIRECT_ANSWER promotion + RATIONALE/scope demotion | PROVEN (ORACLE) |
| Non-oracle LQO classifier improves P@3 modestly (+.03 bench/+.025 hold) but with regressions | EXPERIMENTALLY SUPPORTED (weak) |
| Deterministic RATIONALE suppression is unsafe (relevant-chunk regressions) | PROVEN (H08/H17) |
| A trained second-stage model can currently realize oracle gains | HYPOTHESIS (unsupported: data too small) |
| The answerability mechanism is fundamentally able to help | ORACLE / UPPER-BOUND |
| Whether any production-viable answerability model exists | UNKNOWN (needs more diagnostic data) |

## Artifacts (all new, isolated under `evaluation/experiments/step6/`)

- `run_step6.py` — reproducible experiment (loads frozen data, reorders top-10, α-sweep, writes
  `results/*_order.jsonl` + `results/summary.json`).
- `analyze_step6.py` — affected-subset metrics, per-question deltas, regression/distractor-class
  analysis (`results/analysis.json`).
- `results/` — all order files, `summary.json`, `analysis.json`.
- `README.md` — how to reproduce.
- No frozen file, label, baseline, or production code was modified.