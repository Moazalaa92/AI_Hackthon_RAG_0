# Generation Failures (Phase 8.5)

Analysis of the 14 non-correct answerable cases (11 partially correct + 3
incorrect) plus a note on the 3 deliberate negatives. Evidence presence is
taken from the frozen retrieval labels (`relevant` in the frozen labeled
files); every answerable question had relevant evidence in its Top-10.

## Summary of patterns

| Pattern | Questions | Count |
|---|---|---|
| A. Faithful-but-beyond-reference (model added grounded detail the curated reference omitted) | H01, H17, H21, H22, Q06, Q08, Q15, Q18 | 8 |
| B. Genuine omission of reference points (answer correct but incomplete) | H07, H18, H23, H25 | 4 |
| C. Conflation of two closely-related questions | H09 | 1 |
| D. Over-abstention (false refusal on an answerable question) | H12 | 1 |

All 14 were **generation-stage** failures: relevant evidence was present in the
frozen Top-10 for every case. There were **zero retrieval failures**.

---

## Pattern A — Faithful-but-beyond-reference (8 cases)

The model produced a **fully grounded** answer that included additional detail
present in the retrieved context but not in the curated reference answer. The
correctness judge penalized these as `partially_correct`/`incorrect` because
the reference was narrower than the context. Grounding was judged `grounded`
in every one of these cases.

Examples:

- **H21** (`incorrect`): answer lists GLUT1 deficiency, pyruvate dehydrogenase
  deficiency, infantile spasms, Doose, Dravet, Lennox–Gastaut and
  drug-resistant epilepsy. Judge: *"adds several other conditions and
  drug-resistant epilepsy, which are not present in the reference"*; grounding
  judge: *"Every claim is directly supported by the retrieved context,
  specifically chunk 2 which lists the exact syndromes"*. The reference answer
  (curated to "such as GLUT1 deficiency syndrome") was too narrow; the model's
  answer matched the actual guideline text more fully than the reference.
- **Q06** (`partially_correct`): answer adds autism, structural abnormality,
  cognitive decline; grounding judge confirms chunks 1–2 support every claim.
- **H01** (`partially_correct`): answer adds 12-lead ECG, neuroimaging and
  sepsis checks as part of initial assessment; all present in the context and
  part of the recommendation. Reference was narrower.
- **Q08** (`partially_correct`): answer adds the infantile-spasms 24-hour
  urgent referral (6.3.1) alongside the 3.1.4 criteria, all from context.
- **Q15** (`partially_correct`): answer adds "compelling reasons" to the
  reproductive-risks exception; the context (p.160) says "or reproductive
  risks do not apply", so the extra qualifier is a small grounding-preserving
  paraphrase.
- **H17**, **H22**, **Q18**: similar small additions (e.g. side effects
  discussion, "if appropriate", "prescribed antiseizure medication") — all
  grounded per the grounding judge.

**Interpretation:** the correctness judge compares strictly against the
curated reference, which is narrower than the context. These labels reflect
**reference-answer coverage**, not model hallucination: the grounding dimension
(100% grounded) shows the model stayed inside the evidence. Whether this is a
"failure" depends on whether extra grounded detail is acceptable for the
application.

---

## Pattern B — Genuine omission of reference points (4 cases)

The answer is correct and grounded but **incomplete**: it omits one or more
reference points that the retrieved evidence supports.

- **H07** (`partially_correct`, `partial`): covers tailoring, involving
  children, support groups, self-management, but omits that information should
  be **repeated at different time points** (2.1.x), a key recommendation in
  the evidence.
- **H18** (`partially_correct`): lists levetiracetam and sodium valproate but
  omits the reference's caution that sodium valproate should be used with
  caution after discussing risks and benefits (6.5.x evidence text).
- **H23** (`partially_correct`, `partial`): covers night monitors and
  medication adherence but omits **discussing individual risk of
  epilepsy-related death from diagnosis** and **agreeing ways to reduce risk**
  (10.2.x).
- **H25** (`partially_correct`, `partial`): covers individualised planning but
  omits the reference to the **NICE transition guideline** and **repeating
  information at different time points** (11.2.x).

**Interpretation:** the model summarizes the most salient evidence but
sometimes drops secondary points of multi-part recommendations. This is a
completeness weakness, not a grounding or factual error.

---

## Pattern C — Conflation of two closely-related questions (H09)

- **Question:** "What type of treatment strategy does the guideline recommend
  when starting antiseizure medication?"
- **Answer:** "The guideline recommends using a single antiseizure medication
  (monotherapy) when starting treatment."
- **Reference:** "Develop an individualised antiseizure medication treatment
  strategy with the person… (4.1.1)."
- **Relevant evidence:** chunk rank 7 (p.39) contains 4.1.1 verbatim
  ("Develop an individualised antiseizure medication treatment strategy…").
  The Top-10 also contains monotherapy text (4.1.3, chunk rank 1–2).
- **What went wrong:** the model answered the *closely-related* question
  (monotherapy, which is the answer to H10) instead of the *asked* question
  (individualised strategy, 4.1.1). Both topics are in the context; the model
  latched onto the more frequent/salient one (monotherapy appears in ranks
  1–2).
- **Classification:** generation failure (evidence present; the model selected
  the wrong recommendation among similar ones).

---

## Pattern D — Over-abstention / false refusal (H12)

- **Question:** "What MHRA safety advice should be followed when switching
  between different manufacturers' versions of an antiseizure medication?"
- **Answer:** "The provided context does not contain the specific MHRA safety
  advice… the answer was not found in the provided documents."
- **Reference:** "Follow the MHRA safety advice on switching between different
  manufacturers' products of a particular antiseizure medication (4.3.1)."
- **Relevant evidence:** the frozen retrieval marked a chunk relevant for H12;
  the context contains the "Follow the MHRA safety advice on switching between
  different manufacturers' products" recommendation.
- **What went wrong:** the model judged the question unanswerable because the
  context does not spell out *what* the advice says — but the question asks
  *what advice should be followed*, whose answer is the recommendation itself
  ("follow the MHRA safety advice on switching"). The refusal is
  understandable (the context points to external MHRA guidance rather than
  restating it), and the grounding judge agreed the context does not detail the
  advice — but against the reference, the answerable response was
  "follow the MHRA safety advice".
- **Classification:** generation failure — over-cautious abstention on a
  question the reference considers answerable. Borderline; this is the kind of
  case where "not found" may be acceptable behavior for a conservative
  assistant.

---

## Deliberate negatives (Q20, H26, H27) — no failure

All three negatives were judged `appropriate_refusal` (3/3). Example reasons:
- Q20: "correctly stated that the recommended starting dose was not found… as
  the context only discusses levetiracetam as a treatment option without
  specifying dosages."
- H27: "correctly stated that the answer was not found in the provided
  documents, without fabricating any cost information."

No inappropriate answer on any negative case.

---

## Overall reading

- **No hallucination:** grounding was `grounded` on 44/44 (100%). The model
  never invented facts outside the retrieved context.
- **Good abstention:** 3/3 deliberate negatives correctly refused.
- **Main weaknesses:** (1) occasional incompleteness on multi-part
  recommendations (4 cases), (2) conflation of closely-related questions
  (1 case), (3) over-cautious refusal (1 case). The largest group of
  "non-correct" labels (8) is a reference-answer-coverage artifact: the model
  stayed grounded but was more complete than the curated reference, so the
  strict correctness judge under-rated it.