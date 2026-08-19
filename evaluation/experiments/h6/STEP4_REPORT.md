# H6 Offline Experiment — STEP 4 Report

Standalone experiment. Nothing in the production pipeline was modified. All
variants operate as a **pure post-reranking reorder** on the frozen
`reranked_hybrid_800_100_candidates.jsonl` / `holdout_v1_reranked_hybrid_candidates.jsonl`
files. No chunk is ever deleted; candidate-pool recall stays exactly 100% (verified:
615/615 benchmark, 858/858 holdout). The heuristic uses **only** chunk text,
question text, page label and reranker score — never the `relevant/not_relevant`
labels (those are used for evaluation only).

Run with: `.venv/bin/python evaluation/experiments/h6/run_h6.py`

---

## 1. Metadata available (STEP 4A)

Inspected the frozen candidate records. No `section`/heading field exists
(`section: None` on all 1473 records). Signals must therefore be **text-based**.
Available per record: `chunk_text`, `question_id`, `question_text`, `page`,
`page_label`, `dense_rank`, `dense_score`, `bm25_rank`, `bm25_score`,
`fusion_score`, `reranker_rank`, `reranker_score`, `retrieved_by`.

Verified the frozen Top-10 file == top-10 by `reranker_rank` in the candidates
file (0 mismatches), so baseline metrics reproduce exactly.

## 2. Distractor signals (STEP 4B/4C)

Six text signals were defined and their **relevant-rate** measured on the frozen
labels (validation only; the heuristic itself is label-free):

| Signal | Pattern | Fires | relevant-rate | Verdict |
|---|---|---|---|---|
| F1_toc | `\.{6,}` (TOC dot leaders) | 8 | 0.00 | **safe, strong** |
| F2_research | `recommendation for research` | 30 | 0.00 | **safe, strong** |
| F3_ta | `nice technology appraisal guidance` | 4 | 0.00 | **safe, strong** |
| F4_affect | `how the recommendations might affect practice` | 52 | 0.04 | unsafe alone |
| F5_evidx | `evidence review X:` at line start | 32 | 0.03 | unsafe alone |
| F6_why | `why the committee made these recommendations` | 44 | 0.05 | unsafe alone |

F4/F5/F6 and the scope-qualifier signal were **tested and rejected**: relevant
chunks share the same surface text (e.g. Q10's c204 is the *relevant* rank-3
answer and fires F5_evidx; Q11's relevant chunk c205 contains `add-on treatment`;
H08's relevant c292 fires F6_why). Using them demoted relevant evidence.

**D-class (broad semantic neighbors) cannot be detected robustly without another
model.** The cross-encoder already scores these chunks high; distinguishing "EEG
chunk for an MRI question" from a genuinely useful neighbor requires query-aware
semantic entailment, i.e. a second model. We state this explicitly rather than
invent a fragile heuristic. H6 therefore fixes only the F-class (index/rationale)
distractors that are measurable and 0%-relevant.

## 3. H6 variants (STEP 4D)

| Variant | Algorithm | Demotes | Never demotes | Changes rank 1? | Removes chunks? |
|---|---|---|---|---|---|
| H6-A | **F1/F2/F3 only**: `score -= 3.0` on F1_toc or F2_research, `-= 2.0` on F3_ta | Only chunks containing a TOC dot-leader line, `recommendation for research`, or `NICE technology appraisal guidance` | Everything else, incl. all relevant Q09/Q10/Q11/Q16/H08 chunks | No (no benchmark/holdout rank-1 is F1/F2/F3) | No |
| H6-B | H6-A + limited diversity: if 2+ F-flagged unprotected chunks share a page, demote the extra by 1.0 | (same as A) + additional F-flagged chunk on a duplicated page | protected (evidence-bearing) chunks | No | No |
| H6-C | H6-B + scope-mismatch: `score -= 0.75` when chunk has `second-line`/`with other seizure types`/`add-on treatment`/etc. but question doesn't | (same as B) + scope-narrowed chunks | protected chunks (for B); scope applies to all | Yes (H09/H17 holdout) | No |

### Protection rules (STEP 4E) — applied in all variants
- Never delete a chunk (reorder only) → recall stays 100%.
- Protected = numbered recommendation (`^\d+\.\d+`) or (action verb AND medicine name). Protected chunks are never demoted by F4/F5/F6 (not used in H6-A) or by diversity.
- H6-A's three signals have **0% relevant-rate** on the full pool, so no evidence-bearing chunk can be demoted by them.
- The heuristic is built from text patterns only; labels were consulted once to *validate* signal precision, not to construct the ranking policy.

## 4. Results (STEP 4F) — benchmark

| Metric | Baseline | H6-A | H6-B | H6-C |
|---|---|---|---|---|
| P@1 | 0.850 | **0.850** | 0.850 | 0.850 |
| P@3 | 0.533 | **0.550** | 0.550 | 0.550 |
| P@5 | 0.390 | **0.410** | 0.410 | 0.410 |
| MRR | 0.879 | **0.879** | 0.879 | 0.879 |
| Hit@3 | 0.900 | 0.900 | 0.900 | 0.900 |
| Hit@5 | 0.950 | 0.950 | 0.950 | 0.950 |

## 5. Per-question deltas (benchmark, H6-A)

| Q | base P@3 | new P@3 | delta | base P@5 | new P@5 | delta | first-rel rank |
|---|---|---|---|---|---|---|---|
| Q12 | 0.667 | 1.000 | **+0.333** | 0.600 | 0.600 | 0 | 1 |
| Q13 | 0.667 | 0.667 | 0 | 0.400 | 0.600 | **+0.200** | 1 |
| Q18 | 0.333 | 0.333 | 0 | 0.200 | 0.400 | **+0.200** | 1 |
| all others | unchanged | unchanged | 0 | unchanged | unchanged | 0 | unchanged |

Improved: 3 · unchanged: 17 · worsened: **0**.

## 6. Holdout (STEP 4H)

| Metric | Baseline | H6-A | H6-B | H6-C |
|---|---|---|---|---|
| P@1 | 0.815 | 0.815 | 0.815 | 0.815 |
| P@3 | 0.580 | 0.580 | 0.580 | 0.580 |
| P@5 | 0.452 | **0.467** | 0.459 | 0.474 |
| MRR | 0.870 | 0.870 | 0.870 | 0.870 |

Per-question (H6-A): H18 +0.200 P@5, H22 +0.200 P@5; **0 regressions**. H6-C
(scope) introduces holdout regressions H09, H17 (P@3 0.667→0.333) and H23 (P@5
0.400→0.200) — rejected.

## 7. Regression analysis (STEP 4G)

- **H6-A / H6-B: zero regressions on either set.** No benchmark or holdout
  question decreases in P@3 or P@5.
- **H6-C regressions (why it is rejected):**
  - **H09** (P@3 0.667→0.333): scope penalty demoted relevant evidence chunks that
    legitimately mention `second-line`/`add-on`.
  - **H17** (P@3 0.667→0.333): same mechanism.
  - **H23** (P@5 0.400→0.200): relevant chunk mentions `recommendation for research`-adjacent phrasing and got demoted by F5/F2 interactions.
  - **Q16 (earlier iteration):** broad scope `in children/in adults/under N years`
    demoted relevant chunks (c316/c442) — the reason the scope signal was narrowed,
    and then rejected entirely.

Conclusion: adding F4/F5/F6 or scope signals trades new ranking errors for
aggregate gains; only the F1/F2/F3 core is monotone-safe.

## 8. Does it fix F/D distractors? (STEP 4I-8)

- **F (index/rationale): yes, partially.** Every H6-A win is a genuine F-class
  distractor demotion: Q12's c267 (NICE TA guidance), Q13's F3_ta/F2_research
  chunks (p102/p103), Q18's c429 (`Other recommendations for research`), H18's
  F2_research chunk, H22's F2_research chunk. In each case a relevant chunk
  entered top-5 where a 0%-relevant index chunk previously blocked it.
- **F not fixed (remains):** Q09 (F4_affect/F5_evidx at rank 1-2) and Q11 (lexical
  scope) are **not** fixed — deliberately, because fixing them requires demoting
  signals that also appear in *relevant* chunks (Q10 c204, Q11 c205/c212). That is
  the correct conservative trade-off.
- **D (broad semantic): not addressed** — requires another model (stated above).

## 9. Does it damage legitimate multiple-evidence cases? (STEP 4I-9)

**No.** H6-A demotes only chunks with a 0%-relevant text marker. Multi-evidence
questions (Q03, Q10, Q15, Q16, H04, H07, H14, H20, H25) are untouched (P@3/P@5
unchanged). The diversity variant (H6-B) was checked specifically against this
concern: it adds no benchmark gain and slightly *hurts* holdout P@5, so it is not
recommended.

## 10. Success criteria (STEP 4I)

| Criterion | H6-A |
|---|---|
| P@3 improves meaningfully | +0.017 benchmark (0.533→0.550) — small but strictly positive |
| P@5 improves | +0.020 benchmark, +0.015 holdout — positive both sets |
| P@1 not decreased | 0.850 → 0.850 ✓ |
| MRR not decreased | 0.879 → 0.879 ✓ |
| Holdout confirms direction | ✓ (P@5 up, no regression) |
| Recall stays 100% | ✓ |
| No new ranking errors | ✓ zero per-question regressions |

**Classification: B — Small/uncertain improvement → needs further investigation.**

Rationale: H6-A is strictly non-damaging (0 regressions, recall preserved, P@1/MRR
flat) and improves P@3/P@5 on the benchmark and P@5 on holdout. But the gains are
small (+1.7% P@3 on 20 questions, i.e. 1 question), Q11/Q09 remain unfixed, and
the D-class is untouched. This is not a "strong improvement" (A): the headroom at
the ceiling is ~+0.20 P@3 and H6-A captures only +0.017 of it. It IS safe and
could ship as a low-risk post-processing step, but alone it does not solve the
retrieval ranking problem.

### Final recommendation
Keep the frozen pipeline as canonical. **Do not adopt H6 as the retrieval
solution.** If anything, H6-A's F1/F2/F3 suppression could be considered a
*candidate* for production *only* after further investigation of the two bigger
levers it does not touch: (1) a second-model (NLI/entailment) specificity signal
for the D-class, and (2) the Q11-style lexical-scope problem. That is STEP 5
territory. No production change is made in this phase.