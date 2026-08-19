# Generation prompt comparison: v1 vs v2 (single controlled experiment)

- Date: 2026-08-18
- Dataset: `evaluation/generation_dataset_v1.json` (47 questions; 44 answerable, 3 negatives Q20/H26/H27) — unchanged.
- Retrieval: frozen hybrid pipeline Top-10 (dense + BM25, cross-encoder rerank), read-only inputs.
- Model: `deepseek/deepseek-v4-flash`, temperature 0.
- Judge: `src/generation_judge.py` — unchanged between runs (4 non-leaking judges, `max_tokens=2000`, 3 retries).
- Artifacts: v1 = `evaluation/results/generation_v1{,_judged}.jsonl`, v2 = `evaluation/results/generation_v2{,_judged}.jsonl`.
- Only experimental variable: the generation system prompt in `src/generation.py` (`GROUNDING_SYSTEM_PROMPT`).

## 1. Original prompt (v1) behavior

The v1 prompt had 4 short rules (use only the provided context; refuse if the context is insufficient; do not invent; be concise). It produced a correct-but-incomplete answer profile:

- 68.2% strictly correct, 25.0% partially correct, 6.8% incorrect (44 answerable).
- 100% grounded, 88.6% complete.
- Refusal 3/3 appropriate; 0 retrieval failures; 14 generation-stage failures.

Observed failure patterns (from `evaluation/generation_failures.md`):

- **Pattern A — multi-part omission:** H07, H18, H23, H25 answered the core of the question but dropped important components/qualifying conditions (e.g. H07 omitted that information should be repeated at different time points; H18 omitted the sodium valproate caution; H23 omitted discussing individual risk from diagnosis; H25 omitted referencing NICE + repetition in accessible format).
- **Pattern B — neighboring-recommendation conflation:** H09 answered "single antiseizure medication (monotherapy)" (4.1.3) where the reference expects an "individualised treatment strategy" (4.1.1) — both were in the context.
- **Pattern C — over-abstention:** H12 refused ("answer not found") although the context explicitly contains the MHRA safety advice to follow.
- **Pattern D — faithful-but-beyond-reference (8 cases):** H01, H17, H21, H22, Q06, Q08, Q15, Q18 gave grounded extra detail beyond the curated reference — not hallucinations, and correctly preserved in v2.

## 2. The change made

Replaced the 4-rule v1 system prompt with a 5-rule v2 prompt that preserves the original grounding contract and adds:

1. **Grounding** (unchanged core): "Ground every claim in the provided context only. Do not use outside knowledge and do not invent missing information."
2. **Exact-question selection (new):** "When the context contains several closely related recommendations, distinguish them and answer the one the question asks about — do not substitute a nearby recommendation merely because it is semantically similar." (targets Pattern B / H09)
3. **Multi-part completeness (new):** "When the question asks for multiple items, conditions, criteria, steps or components, include the important components that the context supports, and do not omit important qualifying conditions." (targets Pattern A / H07, H18, H23, H25)
4. **Refusal nuance (new):** "Only refuse when the context genuinely does not contain enough information to answer. If the context directly contains the answer, answer from that evidence even if the document adds no further explanation." (targets Pattern C / H12)
5. **Conciseness/relevance (kept, reworded):** "Be concise and directly relevant. Do not add tangential information just because it appears in the context."

The `build_prompt` user message, `build_context`, and `generate_answer` were left unchanged; the judge config and evaluation methodology were untouched.

## 3. Why this change

The v1 baseline's failures were almost all generation-stage and split across three coherent weaknesses (multi-part omission, conflation of neighboring recommendations, over-abstention). Each weakness maps to a missing or ambiguous instruction in the v1 prompt. The goal was a single minimal prompt that keeps the strict grounding contract (which was already perfect) while telling the model how to select the exact recommendation, preserve important components, and reserve refusal for genuinely unsupported questions — without introducing chain-of-thought, reasoning exposure, or a long rule list.

## 4. v1 vs v2 metrics (44 answerable)

| Metric | v1 | v2 | delta |
|---|---|---|---|
| Strictly correct | 30/44 (68.2%) | 33/44 (75.0%) | +3 |
| Partially correct | 11/44 (25.0%) | 9/44 (20.5%) | −2 |
| Incorrect | 3/44 (6.8%) | 2/44 (4.5%) | −1 |
| Grounded | 44/44 (100%) | 44/44 (100%) | 0 |
| Complete | 39/44 (88.6%) | 40/44 (90.9%) | +1 |
| Complete-partial | 4/44 (9.1%) | 1/44 (2.3%) | −3 |
| Incomplete | 1/44 (2.3%) | 3/44 (6.8%) | +2 |
| Abstention appropriate | 3/3 | 3/3 | 0 |
| Generation failures | 14 | 11 | −3 |
| Retrieval failures | 0 | 0 | 0 |

Net: strictly-correct rate rose 68.2% → 75.0%, grounding held at 100%, refusal behavior unchanged.

## 5. Case-by-case comparison (targeted weaknesses)

| Case | v1 | v2 | Verdict |
|---|---|---|---|
| H07 (multi-part: info formats/repetition) | partial/partial | correct/complete | **Improved** — repetition now included |
| H09 (conflation 4.1.3 vs 4.1.1) | incorrect/partial | incorrect/incomplete | **Not fixed** — still answers monotherapy; judge still flags omitted individualised strategy |
| H12 (over-abstention MHRA) | incorrect/incomplete | correct/incomplete | **Fixed** — now states MHRA safety advice |
| H18 (multi-part: sodium valproate caution) | partial/complete | correct/complete | **Improved** |
| H23 (multi-part: individual risk discussion) | partial/partial | partial/incomplete | **Not fixed** — still omits discussing individual risk from diagnosis and agreeing ways to reduce risk |
| H25 (multi-part: NICE + repetition) | partial/partial | correct/complete | **Improved** |

3 of the 4 multi-part cases improved; the over-abstention case is fixed; the conflation case (H09) and one multi-part case (H23) remain.

## 6. Grounding change

None — 44/44 grounded in both runs. The v2 additions (first-line meds in H17, emergency/benzodiazepine steps in H20, nurse roles in H24) are all directly supported by the retrieved context per the grounding judge; they were penalized by the *correctness* judge only for being beyond the curated reference (same Pattern D as v1). No v2 answer introduced out-of-context or invented content.

## 7. Refusal change

None — 3/3 appropriate refusals (Q20, H26, H27) in both runs, 0 inappropriate answers. The relaxed refusal rule did not cause the negatives to be answered.

## 8. New regressions introduced by v2

- **H17, H20, H24:** correct/partial → flipped to partially-correct because the answer now includes additional *grounded* detail beyond the reference (Pattern D). Not hallucinations; a side effect of the completeness instruction. No grounding harm.
- **Q08:** completeness complete → partial — the v2 answer dropped the "unilateral structural lesion" and "deterioration in behaviour/speech/learning" criteria. A genuine conciseness-driven omission to watch.
- **Q17:** correct → partially-correct with nearly identical answer text — judge variance (v2 judge penalized the omitted "stop multiple medicines one at a time", which the v1 answer also lacked but was not penalized for).

## 9. Keep/reject decision

**Keep the v2 prompt.**

Per the decision rule: it improves the demonstrated weaknesses (H07, H12, H18, H25 fixed; strictly-correct rate +6.8pp; generation failures 14 → 11) without harming grounding (100%) or refusal (3/3 appropriate). No new grounding or refusal problems were introduced. The remaining errors (H09, H23) are the same residual weaknesses, not new ones. One experiment only — no further prompt optimization.

## 10. Remaining failure patterns and Phase 9 readiness

- **Still failing:** H09 (recommendation conflation — the two candidate recommendations coexist in the context and the model chooses the wrong one; the exact-question rule did not resolve it) and H23 (multi-part omission of the "discuss individual risk from diagnosis / agree ways to reduce risk" items).
- **Watch items:** Q08 conciseness-driven omission; H17/H20/H24 grounded-but-beyond-reference additions (correctness judge penalizes these; a future reference-quality pass or judge refinement could reduce this noise, but that is outside this experiment's scope).
- **Phase 9 readiness (Sources/citations):** ready. `build_context` already preserves chunk_id/page/page_label/source/rank metadata through generation, and `generate_answer` consumes it unchanged; `evaluation/results/generation_v2.jsonl` records the retrieved-chunk ids/pages/ranks alongside each answer, so answer-to-source mapping is available for the citations phase without re-running retrieval or generation.