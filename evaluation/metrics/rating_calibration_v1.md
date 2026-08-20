# Rating calibration v1 — what a support band is allowed to claim

Reproduce with (offline, zero LLM calls):

```bash
python scripts/calibrate_rating_bands.py
python scripts/calibrate_rating_bands.py --sweep
```

Inputs are frozen artifacts only: `generation_v2_judged.jsonl` (correctness
labels), `citations_v1.jsonl` (claims, citations, validation), the frozen
reranked Top-10 files, and the hybrid candidate files (for `retrieved_by`).
The script does not reimplement the band rule — it builds the same `Answer`
and `SafetyResult` the API builds and calls `src.rating.rate_answer`.

## Result at the provisional thresholds (`T_TOP1=1.0`, `T_MARGIN=0.5`)

47 judged questions, 44 answerable, baseline correct rate **33/44 = 0.750**.

| band | answerable | correct | partial | incorrect | correct rate | 95% Wilson CI | negatives |
|---|---:|---:|---:|---:|---:|---|---:|
| high | 43 | 32 | 9 | 2 | 0.744 | [0.598, 0.851] | 0 |
| medium | 1 | 1 | 0 | 0 | 1.000 | [0.207, 1.000] | 0 |
| low | 0 | 0 | 0 | 0 | – | – | 0 |
| refused | 0 | 0 | 0 | 0 | – | – | 3 |

The three negatives (Q20, H26, H27) are all classified `refused`; no
answerable question is refused.

## The finding: the bands do not separate correctness

`high` scores 0.744 against a 0.750 baseline — the band is indistinguishable
from "any answer at all". The `--sweep` over `T_TOP1 ∈ {-6…6}` and
`T_MARGIN ∈ {0, 0.25, 0.5, 1, 2, 4}` finds no separating threshold pair:

- For every `T_TOP1 ≤ 4.0`, the `medium` band is *more* accurate than `high`
  (gap −0.26 to −0.04), i.e. the signal points the wrong way.
- The largest positive gap anywhere in the grid is `T_TOP1=6.0`:
  `high` 29/38 = 0.763 vs `medium` 4/6 = 0.667, gap +0.096 with 6 cases in the
  weaker band — well inside the noise of the CIs above.
- `low` is empty at every threshold, because citation coverage is 1.0 and
  `citations_valid` is true for all 47 questions.

Two independent reasons for this, both visible in the artifacts: cross-encoder
top-1 scores measure whether *a* passage matches the question, not whether the
generated text used it correctly; and the citation validator already gates the
same failure mode the `low` band was meant to catch, so it never fires on a
successfully generated answer.

## Consequence for the product

1. **Do not present `high` and `medium` as accuracy tiers.** They are not.
   The UI shows one supported state for both, worded as an evidence property
   ("every claim cited to a guideline page"), never as a per-answer accuracy.
2. **The only honest per-answer number is corpus-level**: on these 44
   answerable questions, answers of the supported kind were judged strictly
   correct 33/44 = 75.0% (95% CI [0.606, 0.854]), partially correct 20.5%
   (9/44), incorrect 4.5% (2/44), and grounded in the retrieved text 44/44.
3. **Keep the `low` and `refused` rules.** `low` never fires on this eval set
   but is a live guard: citation selection does fail in production (11 of 47
   generation attempts failed at least once in the v2 run), and `refused`
   is the safety and insufficient-evidence path, correct on 3/3 negatives.
4. **Thresholds stay provisional and unused for display.** They remain
   overridable arguments so a future, larger judged set can be tested for
   separation with the same script. Tuning them now would only pick noise.

Not calibrated here: `dual_retrieval` and `page_agreement_top5` are recorded
as signals and returned by the API, but no threshold on them was validated —
the same 44-question set is too small to fit a second discriminator on.
