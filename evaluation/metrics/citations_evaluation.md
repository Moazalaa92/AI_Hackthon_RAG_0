# Citation Evaluation (Phase 9)

Measurement of the claim-level citation layer built in Phase 9. It answers:
"does the answer link every important claim to the exact retrieved evidence?"

## Methodology

- **Pipeline measured:** frozen retrieval Top-10 → generation (v2, frozen) →
  `src/sources.py` claim splitting + deterministic candidate pre-filter +
  LLM chunk selector → citations → `src.sources_judge.py` support judge.
- **Generation inputs:** `evaluation/results/generation_v2.jsonl` (47 records;
  already embeds the frozen Top-10 retrieved chunks per question). Read-only.
- **Citation artifacts:**
  - `evaluation/results/citations_v1.jsonl` — 47 questions, 92 claims, 130
    citations, with build-time deterministic validation.
  - `evaluation/results/citations_v1_judged.jsonl` — per-citation support
    labels.
  - `evaluation/metrics/citations_evaluation.json` — aggregate metrics.
- **Judge note (important):** the support labels in this run use the
  **deterministic fallback** in `src/sources_judge.py` because the OpenRouter
  account has zero credits (HTTP 402 on every LLM call). The LLM judge is the
  primary path and is preserved; the deterministic proxy is used only when the
  API is unavailable. Deterministic verdicts are tagged `judge=deterministic`
  and are **more lenient** than the LLM judge (they measure token overlap, not
  semantic entailment). When credits are restored, re-running
  `scripts/evaluate_citations.py` (resumable) will replace the labels.

## Three dimensions measured

| Dimension | Definition | Method |
|---|---|---|
| **Coverage** | Does every important factual claim have a citation? | Deterministic: claims with ≥1 citation / answerable claims. Refusal/no-evidence claims (e.g. "the answer was not found") are excluded by design. |
| **Support** | Does the cited chunk actually support the claim? | LLM evaluator (primary) or deterministic token-overlap proxy (fallback), per citation. |
| **Traceability** | Can we follow the citation to the exact source text/page/chunk? | Deterministic: every citation's chunk_id ∈ retrieved Top-10, supporting_text == exact chunk_text, page/page_label/source copied from the chunk record. |

## Results

| Metric | Result |
|---|---:|
| Questions | 47 / 47 |
| Claims (all) | 92 |
| No-evidence claims (uncited by design) | 1 |
| **Coverage (answerable claims)** | **88 / 88 = 100%** |
| Citations | 130 |
| **Support — supported** | **111 / 130 = 85.4%** |
| Support — partially_supported | 19 / 130 = 14.6% |
| Support — unsupported | 0 / 130 = 0% |
| **Traceability — questions valid** | **47 / 47 = 100%** |
| **Traceability — citations point to a retrieved chunk** | **130 / 130 = 100%** |

Support labels in this table are from the **deterministic fallback judge**
(`judge=deterministic` on all 130 citations) due to the OpenRouter credit
outage. The earlier LLM-judge run (before the outage) measured
**109/130 supported (90.1%), 10/130 partially_supported (8.3%),
2/130 unsupported (1.7%)** — the LLM judge is stricter.

## Example citation (Q01)

- **Question:** "How does the guideline define drug-resistant epilepsy?"
- **Claim:** "The guideline defines drug-resistant epilepsy as epilepsy in
  which seizures persist and seizure freedom is very unlikely..."
- **Citation:** `Q01-C1-C1` → `chunk_412`, page_label 148,
  source `data/pdfs/Project_pdf.pdf`, reranker rank 1, `relevant=relevant`,
  support `supported`.
- **Supporting text (exact chunk text):** "Terms used in this guideline. This
  section defines terms that have been used in a particular way for this
  guideline. The definitions for the epilepsy..."

Every field of the citation was copied from the frozen chunk record; nothing
was invented. Traceability holds because `chunk_412` was in Q01's retrieved
Top-10 and the supporting text is byte-identical to the stored chunk text.

## Limitations

- The deterministic support judge measures lexical token overlap, not semantic
  entailment, so its `supported` labels are looser than the LLM judge's. The
  LLM judge (90.1% supported) is the more reliable measurement and should be
  re-run when API credits are available.
- Coverage is measured at the claim level. A claim can be a long sentence
  containing two facts; the citation points to the supporting text, and a
  human reads it — granularity is deliberately sentence-level.
- 0% unsupported under the deterministic judge does NOT mean zero citation
  errors: it reflects the lenient lexical proxy, not ground truth.