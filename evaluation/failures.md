# Retrieval Failure Analysis (Phase 12.9)

Baseline configuration: `chunk_size=500`, `chunk_overlap=50`
Embedding: `sentence-transformers/all-MiniLM-L6-v2` (Chroma, L2 distance, lower = more similar)
Labels: LLM-as-a-Judge (deepseek-v4-flash, temp 0) — proxy for relevance, not clinical ground truth.

Source artifacts (all read-only):
- `evaluation/dataset.json`
- `evaluation/results/baseline_500_50_top10.jsonl` (frozen retrieval)
- `evaluation/results/baseline_500_50_top10_labeled.jsonl` (labels)
- `evaluation/metrics/baseline_500_50_topk.json` (Top-K analysis)

Note: Generation is not implemented, so every case below is a **retrieval** analysis only.
Any generation note would be out of scope.

---

## Failure 1 — Q07

- Question: Within what time should a child, young person or adult be referred after a first suspected seizure?
- Expected/relevant information: Referral "urgently (for an appointment within 2 weeks)" after a first suspected seizure (NICE NG217 1.1.1).
- Observed retrieval behavior: **First relevant rank: 9**. Relevant chunks: rank 9 only. No relevant chunks in ranks 1–8.
- Retrieved results (top-10, L2 distance, page):
  - rank 1: chunk 46, page 11, dist 0.626 — self-referral instructions after a *further* seizure.
  - rank 2: chunk 150, page 38, dist 0.675 — immediate referral for specific scenarios (deterioration, surgery).
  - rank 3: chunk 39, page 9, dist 0.696 — referral to a clinician with expertise; 2-week timing only for seizure *recurrence after remission* (1.1.2).
  - rank 4: chunk 492, page 122, dist 0.696 — emergency management plans, repeated/cluster seizures.
  - rank 5: chunk 149, page 38, dist 0.705 — referral of specific subgroups (under 3 years, myoclonic seizures).
  - rank 6: chunk 140, page 35, dist 0.709 — tertiary referral / infantile spasms context.
  - rank 7: chunk 40, page 9, dist 0.734 — referral discussion continued.
  - rank 8: chunk 452, page 112, dist 0.742 — epilepsy specialist / seizure type referral.
  - rank 9: **chunk 38, page 9, dist 0.760 — RELEVANT** (the answer).
  - rank 10: chunk 56, page 13, dist 0.764 — first-seizure assessment context.
- Rank of the relevant chunk (if found): 9 (within Top-K; retrieved, but ranked last).
- Was the correct chunk outside Top-K? no.
- Relevant chunk (excerpt): "1.1.1 Refer children, young people and adults urgently (for an appointment within 2 weeks) for an assessment after a first suspected seizure:" — note the trailing colon: the chunk is **cut at the boundary** and the sentence continues in the next chunk (chunk 39, which ranks higher at rank 3).
- Higher-ranked distractors: chunks about other referral topics (further seizures, immediate referral for deterioration, recurrence-after-remission timing, specific subgroups). They overlap heavily with the question's keywords ("refer", "seizure", "assessment", "children/young people/adults") but do not answer the 2-week timing for a *first suspected seizure*. Judge reasons confirm: each lacks the first-suspected-seizure time frame.
- Evidence-based diagnosis: The answer chunk (page 9 = expected page) **is retrieved but ranked 9th**. The top-10 distances are flat (0.626–0.764), so ranking is fragile. The answer sits at the **end** of chunk 38, truncated by the 500-char boundary, with its continuation in chunk 39 (rank 3). Distractors on the same page/theme outrank it.
- Observed failure pattern: "right page, wrong chunk ordering" + answer context split across a chunk boundary.
- Suspected reason (hypothesis, not fact): the 500-character chunk boundary truncates the answer sentence, and the truncated answer chunk embeds as less similar than nearby topical distractors; the small embedding model (all-MiniLM-L6-v2) gives a flat, weakly discriminative top-10.
- Possible improvement (hypothesis only): larger chunk_size (800/100) so the full recommendation stays inside one chunk; test in Phase 13.

---

## Failure 2 — Q11

- Question: Which medicine is offered as first-line treatment for absence seizures?
- Expected/relevant information: "Offer ethosuximide as first-line treatment for absence seizures" (NICE NG217 5.3.1).
- Observed retrieval behavior: **First relevant rank: 4**. Relevant chunks: ranks 4 and 7. No relevant chunks in ranks 1–3.
- Retrieved results (top-5 + rank 7):
  - rank 1: chunk 313, page 78, dist 0.441 — rationale discussing "it" (ethosuximide, never named) as monotherapy/add-on; judged not_relevant because it refers to the drug only as "it".
  - rank 2: chunk 305, page 76, dist 0.540 — 5.3.4 first-line for absence seizures **with other seizure types** (qualifier mismatch).
  - rank 3: chunk 302, page 75, dist 0.550 — 5.3.2 **second-line** options (lamotrigine, levetiracetam, sodium valproate).
  - rank 4: **chunk 301, page 75, dist 0.565 — RELEVANT** ("5.3.1 Offer ethosuximide as first-line treatment for absence seizures").
  - rank 5: chunk 322, page 80, dist 0.612 — evidence review discussion of absence-seizure treatments.
  - rank 6: chunk 317, page 79, dist 0.613 — first-line for absence seizures with other seizure types (valproate).
  - rank 7: **chunk 312, page 78, dist 0.623 — RELEVANT** (committee rationale: ethosuximide as first-line for absence seizures).
- Rank of the relevant chunk (if found): 4 (and 7) — within Top-K.
- Was the correct chunk outside Top-K? no.
- Relevant chunk (excerpt): "5.3 Absence seizures … 5.3.1 Offer ethosuximide as first-line treatment for absence seizures." — the answer is the **final line** of chunk 301, at the chunk boundary.
- Higher-ranked distractors: chunks that share the question's exact phrases — "first-line", "offer", "absence seizures" — but describe second-line treatment (5.3.2), absence seizures *with other seizure types* (5.3.4), or the rationale paragraph that never names the drug. High lexical overlap with the question's wording pulls them above the answer.
- Evidence-based diagnosis: This is a **ranking weakness, not a retrieval failure** — two relevant chunks were retrieved (ranks 4 and 7). The answer chunk is outranked by lexically-overlapping distractors that mention "first-line … absence seizures" but in a different sense (second-line / with-other-types).
- Observed failure pattern: "synonyms/qualifiers not distinguished" + answer at chunk end.
- Suspected reason (hypothesis, not fact): the question's phrase "offered as first-line treatment for absence seizures" is nearly identical to non-answer sentences (5.3.2 "offer a choice … as second-line … absence seizures"; 5.3.4 "first-line treatment for absence seizures with other seizure types"); the embedding ranks by surface similarity, so qualifier changes ("with other seizure types", "second-line") are not enough to demote distractors. The answer at the chunk boundary may also weaken its embedding.
- Possible improvement (hypothesis only): test larger chunks / higher overlap so 5.3.1 stands apart from the qualifying sections, and possibly a stronger embedding; compare 500/50 vs 800/100 in Phase 13.

---

## Failure 3 — Q18

- Question: When is more frequent monitoring recommended for a pregnant woman with epilepsy?
- Expected/relevant information: pregnancy is a clinical condition needing closer supervision, i.e. monitoring is recommended (NICE NG217 4.4.3).
- Observed retrieval behavior: **First relevant rank: 4**. Relevant chunks: rank 4 only. No relevant chunks in ranks 1–3 or 5–10.
- Retrieved results (top-5):
  - rank 1: chunk 238, page 60, dist 0.588 — general discussion of monitoring in pregnancy (no explicit "when").
  - rank 2: chunk 229, page 57, dist 0.598 — 4.5.7/4.5.8 monitoring levels in women planning pregnancy (not pregnant/frequent).
  - rank 3: chunk 230, page 58, dist 0.702 — 4.5.9 monitoring and adjusting dosages for pregnant women on certain drugs (does not state the "when").
  - rank 4: **chunk 212, page 53, dist 0.733 — RELEVANT** ("a specific clinical condition needing closer supervision (such as pregnancy or renal failure)").
  - rank 5: chunk 236, page 59, dist 0.764 — baseline levels / research-needs discussion.
- Rank of the relevant chunk (if found): 4 — within Top-K.
- Was the correct chunk outside Top-K? no.
- Relevant chunk (excerpt): "4.4.3 Consider monitoring antiseizure medication levels in people with epilepsy and any of the following: … a specific clinical condition needing closer supervision (such as pregnancy …)".
- Higher-ranked distractors: chunks on pages 57–60 that contain the question's exact keywords — "monitoring", "pregnancy", "antiseizure medication levels" — but discuss monitoring *in* pregnancy (drug levels, dosing) rather than *when more frequent monitoring is triggered*. They share surface vocabulary but not the answer.
- Evidence-based diagnosis: The answer chunk was retrieved (rank 4) but is outranked by three keywords-matching distractors. This is a **ranking weakness, not a retrieval failure**. The answer uses the phrase "closer supervision" where the question says "more frequent monitoring" — a semantic paraphrase, not exact wording.
- Observed failure pattern: "synonyms not matched" (closer supervision vs more frequent monitoring) + topic competition with pregnancy-monitoring chunks.
- Suspected reason (hypothesis, not fact): the answer chunk's wording ("closer supervision", "renal failure") diverges from the question's wording ("more frequent monitoring", "pregnant"), while the distractors match the surface terms closely; the small embedding model ranks surface overlap above semantic equivalence. The answer is also one bullet among several, diluting the chunk.
- Possible improvement (hypothesis only): test a stronger embedding and/or hybrid (keyword + vector) retrieval so the "more frequent monitoring" phrase matches the answer; chunking size change may not help this case.

---

## Sanity check — Q17 (NOT a failure)

- Question: Over what period should antiseizure medication be reduced when a decision is made to discontinue it?
- Expected/relevant information: "For most medicines, this would typically be over at least 3 months; longer for benzodiazepines/barbiturates" (NICE NG217 4.6.4).
- Observed retrieval behavior: **First relevant rank: 2**. Relevant chunks: rank 2 only.
- Retrieved results:
  - rank 1: chunk 242, page 60, dist 0.439 — 4.6.4 lead-in: "If a decision is made to discontinue antiseizure medication, agree a plan…" (no period; judged not_relevant).
  - rank 2: **chunk 243, page 61, dist 0.460 — RELEVANT** ("…reducing their antiseizure medications gradually: For most medicines … at least 3 months…").
  - rank 3: chunk 168, page 43, dist 0.591 — tapering when switching to another drug.
  - rank 4: chunk 155, page 40, dist 0.592 — starting medication in older people.
  - rank 5: chunk 244, page 61, dist 0.639 — discontinue one at a time / seizures recur.
- Rank of the relevant chunk (if found): 2.
- Was the correct chunk outside Top-K? no.
- Evidence-based diagnosis: The answer is retrieved at rank 2, immediately below the direct contextual predecessor (the 4.6.4 lead-in that states a decision to discontinue was made). The rank-1 chunk is the natural section opener and is tightly related; the answer chunk itself is ranked correctly relative to its context. This is **acceptable ranking**, not a retrieval failure. The lead-in/answer pair was split across the chunk boundary (chunk 242 ends mid-sentence into 243), which is a minor chunking artifact, not a retrieval problem.
- Observed failure pattern: none significant; "answer spread across two adjacent chunks" (minor).

---

## Observed failure patterns

- Relevant chunks usually appear at rank 1 (13/20 questions; 16/20 have relevant material in the top 3).
- The few weaker cases (Q07 rank 9, Q11 rank 4, Q18 rank 4, Q17 rank 2) all have the relevant chunk **retrieved within Top-10**; none is a true "no relevant chunk retrieved" failure. Q20 is the only question with no relevant retrieval, and it is the intentional unanswerable case — correctly not a failure.
- Recurring pattern in the weak cases: the answer chunk is outranked by **lexically overlapping distractors** (second-line treatment, absence seizures with other types, monitoring *in* pregnancy) that share the question's surface keywords but not the answer; the small embedding (all-MiniLM-L6-v2) favors surface overlap, and distances in the top-10 are flat.
- Recurring pattern: answers frequently sit **at the end of a 500-char chunk**, sometimes truncated (Q07 trailing colon; Q17 lead-in split), suggesting chunk-boundary effects.
- Recurring pattern: semantic paraphrases are not matched (Q18 "closer supervision" vs "more frequent monitoring"; Q11 qualifiers not distinguished).

---

## Optimization hypotheses for Phase 13

Ranked by expected usefulness based on the observed evidence (hypotheses only — not implemented):

1. **Test larger chunks (e.g., 800/100).** Evidence: Q07's answer sentence is truncated at the 500-char boundary (trailing colon; continuation in chunk 39 which ranks higher), and Q11/Q18 answers sit at chunk ends. Larger chunks would keep recommendations + their context in one chunk.
2. **Test higher overlap (e.g., 500/100 or 600/75).** Evidence: Q17's lead-in and answer split across adjacent chunks; Q07's answer sentence is cut at the boundary. Higher overlap reduces boundary loss.
3. **Consider hybrid (keyword + vector) or a stronger embedding** if chunk-size changes do not fix the "synonyms/qualifiers not matched" pattern (Q18, Q11). Evidence currently favors surface-overlap ranking.
4. **Only if chunking changes are insufficient**, consider reranking. Not indicated by the current evidence alone.

These are candidate experiments only. They are not run and no chunk configuration is endorsed or condemned based on this phase alone.

---

## Verification

Re-run `scripts/inspect_results.py` for Q07, Q11, Q17, Q18 to confirm the documented ranks/scores/pages match the persisted results (performed during Phase 12.9).
