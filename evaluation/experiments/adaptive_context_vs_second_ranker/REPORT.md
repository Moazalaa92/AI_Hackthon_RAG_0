# Idea A vs Idea B — Adaptive Context Depth vs Second Lightweight Ranking Layer

Experiment status: **OFFLINE investigation — both ideas tested, neither shows a meaningful
consistent improvement. Recommend NOT pursuing and NOT combining.** No production changes.

Frozen pipeline untouched. All experimental code under `evaluation/experiments/adaptive_context_vs_second_ranker/`.

---

## 1. Executive summary

Two competing ideas for improving the frozen Phase 8.5/12.5 pipeline were tested **separately**,
per instructions, and only combined if one showed meaningful improvement. Neither did.

| Idea | Hypothesis | Verdict |
|---|---|---|
| **A — Adaptive Context Depth** | Pass all reranked chunks with normalized score ≥ threshold instead of fixed Top-10, so Generation sees only high-confidence evidence (and may include relevant chunks ranked 11+). | **Weak, mixed, not robust.** t=0.70 gave +3 correctness / −1 completeness on a 13-question targeted sample; t=0.85 gave −3 correctness / +1. The reranker score is **not** a reliable confidence signal for what Generation needs — the lowest-scored relevant chunk in the Top-10 is often a critical completeness criterion (Q08 rank-9 norm 0.59). Trimming drops it. |
| **B — Second Lightweight Ranking Layer** | Deterministic second-pass reorder of the frozen Top-10 using `(1−α)·norm_CE + α·norm(secondary_signal)` with zero LLM calls. | **Rejected.** No deterministic signal (BM25 score/rank, dense score/rank, RRF rank, lexical overlap, key-term coverage, scope overlap) yields a consistent gain on both benchmark and holdout. Best cases fix one P@1 but introduce holdout P@1/MRR regressions. `fusion_rank` is degenerate (== frozen order at every α), confirming the sanity check. |

**Recommendation: reject both, do not combine.** This matches the STEP 6 finding that only a
genuinely predictive (non-deterministic) answerability signal — e.g. an LLM usefulness judge or a
well-trained embedding classifier — can improve ordering, and that the frozen Top-10 generation is
already strong (44/44 grounded, 39/44 complete).

---

## 2. Hypotheses

**Idea A.** The fixed Top-10 context passed to Generation contains low-confidence, irrelevant chunks
that (a) waste tokens and (b) distract the generator. Passing all chunks whose per-question
normalized reranker score is above a threshold should (a) remove distractor noise and (b)
potentially surface high-confidence relevant chunks ranked 11+ that the frozen Top-10 never shows
Generation. Primary metrics: correctness, completeness, grounding, abstention, answer length,
per-question regressions. Candidate thresholds: 0.70, 0.75, 0.80, 0.85, 0.90.

**Idea B.** A deterministic second pass can reorder the frozen Top-10 so that a genuinely
*different* signal (BM25, dense, RRF, lexical overlap, key-term coverage, scope overlap) corrects
cross-encoder ranking mistakes, with **zero API cost**. Primary metrics: P@3/P@5 gains with no
P@1/MRR regressions, benchmark + holdout.

---

## 3. Score distribution & normalization necessity (Idea A precondition)

The user explicitly warned not to assume scores are normalized to [0,1] and not to pick 0.8
blindly. Analysis of the frozen `reranked_hybrid_800_100_candidates.jsonl`:

- **Raw reranker scores are NOT [0,1]**: Top-10 scores range **−8.10 … 9.81** (all-candidate-pool
  min −11.30, max 9.81), mean 3.85. A fixed raw threshold is meaningless.
- **Scores are NOT comparable across questions**: per-question Top-10 spreads vary from **1.5 to
  12.9** (Q09 2.28, Q11 2.38, H08 4.77, Q06 12.90, Q14 10.94). A threshold that is aggressive for
  Q09 is trivial for Q14.
- **Normalization is therefore required, per-question.** Two options were tested:
  - *Per-question min-max over the Top-10*: collapses to 1–2 chunks for many questions at ≥0.7
    (too aggressive; loses supporting evidence).
  - *Per-question min-max over the FULL candidate pool*: sensible spread (Q06 → 5 @0.7, Q09 → 11,
    Q11 → 22, Q14 → 4, Q18 → 8). **This normalization was used for the generation run.**
- Max-relative (score/max) was also computed but collapses harder at the same thresholds (Q14/Q18
  → 1 chunk at ≥0.7); min-max over the full pool gives the most informative gradient.

Consequence: threshold semantics are per-question ("keep the top ~30% of this question's score
range"), which is exactly the "adaptive" behavior intended — but it also means the threshold can
never be "safe": a relevant chunk sitting in the bottom 30% of the range is dropped.

---

## 4. Idea A — methodology (targeted generation experiment)

- **Context builder**: for each question, take the FULL reranked candidate pool, compute per-question
  min-max over all pool scores, keep chunks with normalized score ≥ threshold, order by
  `reranker_rank`, cap at 25 chunks. Never empty: keep at least the Top-1.
- **Generator**: frozen `src/generation.generate_answer` (unchanged), `max_tokens=600`, exactly as
  `scripts/run_generation_evaluation.py`.
- **Judges**: frozen `src/generation_judge` with the same leakage control as
  `scripts/evaluate_generation.py` (correctness sees no context; grounding sees no reference).
- **Targeted sample (per user instruction — smallest useful sample, minimal tokens)**:
  - Benchmark difficult cases: **Q06 Q07 Q08 Q09 Q11 Q12 Q13 Q14 Q18**
  - Representative holdout: **H01 H08 H21 H23**
- **Thresholds run**: **0.70** (mild trimming, some expansion) and **0.85** (aggressive trimming).
  0.75/0.80/0.90 were not run because 0.85 already shows the failure mode clearly and tokens are
  budgeted.
- Resumable; Q08 consistently returned empty/truncated answers under adaptive contexts (see §8).

Resulting context sizes (n chunks, at t=0.70 / t=0.85):
Q06 5/2 · Q07 8/3 · Q08 7/6 · Q09 11/7 · Q11 22/10 · Q12 13/5 · Q13 8/2 · Q14 4/2 · Q18 8/2 ·
H01 9/5 · H08 9/1 · H21 11/10 · H23 11/5. (Baseline = 10 for all.)

---

## 5. Idea A — benchmark results (Q06–Q18)

| Question | Baseline correctness | t=0.70 | t=0.85 | Baseline completeness | t=0.70 | t=0.85 |
|---|---|---|---|---|---|---|
| Q06 | partially_correct | **correct** | partially_correct | complete | complete | partial |
| Q07 | correct | correct | correct | complete | complete | complete |
| Q08 | partially_correct | partially_correct | **incorrect** (empty ans) | complete | **partial** | incomplete |
| Q09 | correct | correct | correct | complete | complete | complete |
| Q11 | correct | correct | correct | complete | complete | complete |
| Q12 | correct | correct | correct | complete | complete | complete |
| Q13 | correct | correct | correct | complete | complete | complete |
| Q14 | correct | correct | correct | complete | complete | complete |
| Q18 | partially_correct | **correct** | **incorrect** | complete | complete | complete |

- **t=0.70: 2 correctness improvements (Q06, Q18), 1 completeness regression (Q08).**
- **t=0.85: 0 improvements, 2 regressions (Q08→incorrect, Q18→incorrect), Q06 completeness partial.**
- Grounding: 100% grounded in every cell (both thresholds) — trimming never caused hallucination,
  it caused *omission*.

---

## 6. Idea A — holdout results (H01 H08 H21 H23)

| Question | Baseline correctness | t=0.70 | t=0.85 |
|---|---|---|---|
| H01 | partially_correct | partially_correct | partially_correct |
| H08 | correct | correct | **partially_correct** |
| H21 | incorrect | **partially_correct** | incorrect |
| H23 | partially_correct | partially_correct | **correct** |

- **t=0.70: 1 improvement (H21), 0 regressions.**
- **t=0.85: 1 improvement (H23), 1 regression (H08).**
- All grounded at both thresholds.

---

## 7. Per-question deltas, regressions, failure analysis

**t=0.70 combined (13 questions): +3 correctness (Q06, Q18, H21), −1 completeness (Q08), 0 new
incorrect.** Net positive but small and fragile.

**t=0.85 combined (13 questions): +1 (H23), −3 correctness (Q08, Q18, H08).** Clearly negative.

### The failure mechanism (why Idea A is unsafe)

The reranker's score rank is a poor proxy for *generation necessity*. The canonical generation
failures are completeness-driven: answers omit criteria that live in **lower-ranked relevant
chunks**. The clearest example — **Q08**:

- Frozen Top-10 contains 4 relevant chunks at reranker ranks 1, 3, 4, **9**.
- The rank-9 chunk (norm **0.59**) carries the *"unilateral structural lesion"* and *"deterioration
  in behaviour/speech/learning"* criteria — explicitly called out by both the correctness and
  completeness judges.
- Both t=0.70 (norm ≥ 0.7) and t=0.85 drop it → answer omits two of four criteria. At t=0.85 the
  generator returns an **empty answer** (reproduced 3×), judged `incorrect`.

So the "high-confidence evidence" that adaptive context preserves is exactly the chunks the
cross-encoder is confident about — but the *completeness-critical* chunk is usually the one with the
lowest score. This mirrors the STEP 5 finding that 47% of relevant chunks are NOT DIRECT_ANSWER and
that 5/10 wide-margin failures are driven by low-ranked evidence.

### The improvement mechanism (why t=0.70 helps a little)

Q06 and Q18 improved because trimming removed *distractor* chunks that caused the baseline
generator to add hallucinated extra criteria ("autism, structural abnormality, cognitive decline").
The relevant chunks (Q06 r1/r2/r4, Q18 r1/r7) were all retained at t=0.70. Q21 likewise. This is a
real but **non-robust** benefit: it disappears at t=0.85 and is outweighed by the Q08-style
completeness regressions.

### Expansion case (Q11 → 22 chunks at t=0.70)

Q11's rank-13 relevant chunk (norm 0.83) IS included at t=0.70 and t=0.85. It did not hurt, but Q11
was already `correct` in the baseline, so the expansion added no measurable value. Same for the
12 questions with relevant chunks beyond rank-10 (H07, H08, H10, H16, H19, H21, H23, Q08, Q11,
Q13, Q14, Q16): none of them were baseline generation failures, so expansion had no target to fix.

---

## 8. Answer length, token usage, abstention

- **Average answer length**: baseline 380 chars → t=0.70 360 → t=0.85 282. Shorter answers under
  trimming confirm omission, not improvement.
- **Grounding**: 26/26 grounded across both thresholds (13×2). **Abstention**: never triggered
  (all answerable; N/A).
- **API spend (targeted sample, 13 questions)**: 2 thresholds × (13 generations + 3 judge calls per
  answerable question) ≈ 104 LLM calls + a handful of retries for Q07/Q08/Q11 truncation. This was
  the intended "smallest useful sample"; a full 47-question sweep at 2 thresholds would cost
  ~2×47×4 ≈ 380 calls.
- **Q08 note**: under adaptive contexts (6–7 chunks) the generator returned an empty string 3 times
  and a 267-char truncated answer once. Under the frozen 10-chunk context it returns a full 669-char
  answer. Aggressive context reduction destabilizes this generator for multi-criterion questions.

---

## 9. Idea B — methodology (zero-API deterministic reorder)

- **Scope**: frozen Top-10 pool only (same as STEP 6); never pulls in chunks from beyond top-10.
- **Combined score**: `combined = (1−α)·norm_ce + α·norm(secondary)`, per-question min-max of each
  signal within the Top-10 pool. α ∈ {0.0, 0.1, 0.2, 0.3, 0.5}.
- **Secondary signals** (deterministic, from frozen candidates or computed from question+chunk text):
  `bm25_rank` (1/rank), `bm25_score`, `dense_rank` (1/rank), `dense_score`, `fusion_rank` (1/RRF rank),
  `lex_overlap` (question∩chunk / union), `key_coverage` (question key terms ∩ chunk / key terms),
  `scope_overlap` (non-key terms ∩ chunk / non-key terms). Missing single-source signals → 0.
- **Metrics**: identical to STEP 6 `analyze_step6.py` (P@1/P@3/P@5/MRR/Hit@5 vs frozen labels).
- **Sanity check**: α=0.0 must equal the frozen baseline, and `fusion_rank` (which encodes the RRF
  order the CE was applied to) must be ≈ degenerate.

---

## 10. Idea B — results

Baseline: bench P@1 0.850, P@3 0.533, P@5 0.390, MRR 0.879 · hold P@1 0.815, P@3 0.580, P@5 0.452,
MRR 0.870.

Sanity checks passed: every α=0.0 row == baseline; `fusion_rank` is **exactly** the frozen order at
every α (degenerate, by construction).

Best per-signal results (none is a consistent win):

| Signal / α | bench P@1 | bench P@3 | hold P@1 | hold P@3 | bench MRR | hold MRR | regressions |
|---|---|---|---|---|---|---|---|
| baseline | 0.850 | 0.533 | 0.815 | 0.580 | 0.879 | 0.870 | — |
| bm25_rank α=0.3 | **0.900** | 0.533 | 0.778 | 0.556 | 0.917 | 0.852 | hold P@1 (H05,H17) |
| bm25_score α=0.5 | **0.900** | 0.533 | 0.778 | **0.593** | 0.910 | 0.852 | hold P@1 (H05,H17) |
| key_coverage α=0.2 | 0.800 | **0.550** | **0.852** | 0.568 | 0.858 | 0.889 | bench P@1 (Q07) |
| lex_overlap α=0.2 | 0.850 | **0.567** | 0.815 | 0.556 | 0.883 | 0.864 | hold P@3 (H01,H20) |
| dense_rank α=0.2 | 0.850 | 0.517 | 0.815 | 0.568 | 0.892 | 0.870 | bench P@1 0.65 @α=0.5 |
| dense_score α=0.1+ | 0.850 | 0.533 | ≤0.778 | ≤0.580 | 0.879 | ≤0.852 | worsens with α |

Details:

- **bm25_rank α=0.3 / bm25_score α=0.5** fix the bench P@1 failure on **Q11** (bench P@1 0.85→0.90)
  and improve bench MRR to 0.917, but regress **holdout P@1 on H05 and H17** and hold MRR → 0.852.
  One-step-forward-one-step-back, net zero.
- **key_coverage α=0.2** fixes holdout **H01** (P@1 0.815→0.852, MRR 0.889) but regresses **bench
  Q07** P@1 (0.850→0.800). Not consistent across sets.
- **lex_overlap α=0.2** improves bench P@3 0.533→0.567 (Q06, Q11) but regresses holdout P@3 on
  H01/H20.
- Dense signals monotonically degrade holdout metrics as α grows (dense ranking ≠ relevance for
  these guideline questions); `dense_rank α=0.5` collapses bench P@1 to 0.650.
- **Zero signal fixes the root failure**: the 8/47 questions with a non-relevant chunk at rank-1
  (Q09, Q11, Q20, H01, H12, H21, H26, H27) are not consistently repairable — each candidate signal
  fixes only one of them and breaks another.

**Verdict: Idea B rejected.** No deterministic second-stage signal provides a meaningful,
set-consistent gain, and the best cases trade holdout P@1/MRR regressions for a single bench
improvement. P@3 was never the binding constraint; P@1/MRR regressions are the deal-breaker, exactly
as the user specified.

---

## 11. Cross-idea analysis & combination check

- **Idea A** fixes a *generation* failure mode (distractor-induced hallucination) by removing
  context, but breaks a *completeness* failure mode (low-ranked evidence needed) at the same time.
  The two mechanisms are in direct tension, which is why the threshold has no stable sweet spot.
- **Idea B** operates purely on *retrieval ordering* and cannot add evidence that Generation needs;
  it can only permute the same 10 chunks. Since generation grounding is already 44/44 and the
  remaining failures are omission/hallucination (a context *content* problem, not an order problem),
  reordering cannot fix them — confirmed empirically.
- **Combination is therefore not justified**: neither idea shows a meaningful improvement, so per
  the user's rule ("combine only if one shows meaningful improvement") we do not combine. Any
  combination would inherit Idea B's holdout regressions and Idea A's Q08/Q18 omission risk.

---

## 12. Recommendation & next steps

**Do not pursue Idea A or Idea B; do not change the frozen pipeline.**

The two ideas fail for complementary reasons and together bound the achievable space of
deterministic improvements:

1. Deterministic reordering (Idea B) cannot fix generation failures because ordering does not change
   the evidence set. Confirmed: zero-API signals trade regressions for single fixes.
2. Score-thresholded context trimming (Idea A) removes exactly the low-score evidence that
   completeness depends on. Confirmed: Q08/Q18 regress, generator even returns empty answers under
   aggressive trimming.

Where the signal actually lives (all consistent with STEP 6):

- **Oracle usefulness reordering** (LLM `answer_usefulness` on the Top-10) is the only signal that
  improved P@3 with zero regressions (bench +0.117, hold +0.099) — but the non-oracle LQO classifier
  trained on 254 diagnostic examples was too weak (Pearson ≈ 0.26) to reproduce it. **The lever is
  a better-trained usefulness predictor, not a different deterministic signal.**
- For generation, the concrete, addressable failures are **completeness omissions caused by
  low-ranked relevant chunks** (Q08 r9, etc.). A targeted direction would be a
  **completeness-focused verification pass** in Generation (e.g., a judge-guided "did I cover every
  criterion the context supports" self-check), not context-depth tuning.

Suggested next experiment if the user wants to continue: **retrain the answerability classifier on
the full candidate pool with hard-negative mining + a better embedding, and gate it behind an
abstain rule**, or **add a generation-side completeness verification step**. Both are outside the
scope of this report and would need their own approval.

---

## Appendix — artifacts

- `results/ideaA_t0.70_{bench,hold}_generated.jsonl` / `_judged.jsonl`
- `results/ideaA_t0.85_{bench,hold}_generated.jsonl` / `_judged.jsonl`
- `results/{bench,hold}_{signal}_a{alpha}_order.jsonl` (Idea B reorders)
- `results/summary_ideaB.json` (full Idea B table)
- `results/summary_ideas.json` (per-question Idea A delta table)
- `run_idea_a_generation.py` (resumable generation+judge runner, threshold arg)
- `run_idea_b_second_ranker.py` (zero-API reorder experiment)

Frozen files untouched (git status clean for `src/`, `evaluation/results/`, frozen metrics).
