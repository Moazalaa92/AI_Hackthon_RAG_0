# Citation Failures (Phase 9)

Analysis of citation-level failures found during Phase 9 evaluation. The citation
layer is the last stage of the pipeline, so a citation failure sits *after*
retrieval and generation: the evidence was retrieved, the answer was generated,
and a claim was mapped to a chunk — but the chunk does not fully support the
claim.

## Failure type (the core lesson): relevant ≠ supports

The most common citation failure is not a broken citation but an **over-broad
citation**: the cited chunk is *relevant to the question* (it is in the frozen
Top-10 and topically related) but does **not** support the *specific claim* it
was attached to. Retrieval relevance and citation support are different
dimensions; a relevant chunk can still be the wrong evidence for a particular
claim.

Two concrete cases were identified (both flagged `unsupported` by the LLM
support judge in the earlier LLM-judge run; the deterministic fallback labels
them `partially_supported`, which is more lenient — see below).

---

## Failure 1 — H12, claim C1 → chunk 240

- **Question:** "What MHRA safety advice should be followed when switching
  between different manufacturers' versions of an antiseizure medication?"
- **Claim:** The context states that the committee agreed that "MHRA safety
  advice on switching between different manufacturer's products needs to be
  followed" (Chunk 1) and that clinicians should "Follow MHRA safety advice on
  switching between different manufacturers' products."
- **Cited chunks:**
  - `chunk_134` (page 50, rank 1) — supports the claim (`supported`).
  - `chunk_240` (page 88, rank 3) — **fails support**: the chunk is about
    sodium valproate (risks and benefits discussion for women and girls able to
    have children) and says nothing about switching between manufacturers'
    versions.
- **Observed failure pattern:** "relevant to the question" (sodium valproate
  switching/advice topic is related to antiseizure medication safety advice)
  but **not supporting the claim** (no statement about switching between
  different manufacturers' products). The selector attached an extra,
  topically-adjacent chunk that does not carry the claim.
- **Why it was cited:** the chunk selector operates on token overlap with the
  claim; "MHRA safety advice" / "different manufacturers'" / "switching"
  partially overlap with the chunk's topic (valproate advice), passing the
  pre-filter, and the selector kept it as a supporting citation.
- **Smallest fix (mapping stage):** raise the deterministic pre-filter's
  `min_overlap` and/or cap citations per claim so the selector only keeps
  chunks with strong evidence of the exact claim content, not topical
  adjacency. (Proposed for a future iteration; mapping logic is frozen for this
  evaluation.)
- **Classification:** citation-stage failure (selector over-cited a chunk).
  Retrieval and generation were correct — chunk 134 carries the claim.

---

## Failure 2 — Q16, claim C2 → chunk 20

- **Question:** "When is it acceptable to use topiramate in women and girls of
  childbearing potential?"
- **Claim:** "These conditions include using highly effective contraception,
  having a pregnancy test to exclude pregnancy before starting topiramate, and
  being aware of the risks of topiramate use."
- **Cited chunks:**
  - `chunk_139` (page 51, rank 9) — supports the claim (`supported`): it lists
    the exact conditions (contraception, pregnancy test).
  - `chunk_20` (page 6, rank 5) — **fails support**: it says topiramate can be
    used "unless the conditions of the Pregnancy Prevention Programme are
    fulfilled" but does **not** list those conditions.
  - `chunk_273` (page 99, rank 2) — `partially_supported`: mentions the
    Pregnancy Prevention Programme conditions must be fulfilled, but does not
    enumerate them.
- **Observed failure pattern:** "relevant but not supporting." Chunk 20 is
  genuinely relevant to the question (it states the acceptability rule) but
  does not support the claim's specific content (the *list of conditions*).
  The claim is a list; the chunk references the list without giving it.
- **Why it was cited:** the claim text (contraception / pregnancy test / risks)
  has moderate token overlap with the chunk's topic, so it passed the
  pre-filter; the selector kept it even though it only points to the conditions
  without enumerating them.
- **Smallest fix (mapping stage):** for list-type claims, prefer chunks that
  contain the enumerated items; require support for the claim's substance, not
  just the claim's topic. (Future iteration; mapping is frozen.)
- **Classification:** citation-stage failure (over-broad citation). Retrieval
  was correct (chunk 139 carries the exact list); generation was correct.

---

## Why the deterministic judge shows 0 unsupported

The Phase 9 support labels in the current `citations_v1_judged.jsonl` come from
the **deterministic fallback judge** (`src/sources_judge.py`
`judge_citation_support_deterministic`) because the OpenRouter account has zero
credits (HTTP 402 on every LLM call). The fallback is a lexical token-overlap
proxy:

```text
overlap = |claim_tokens ∩ chunk_tokens| / |claim_tokens|
>= 0.66 -> supported; >= 0.33 -> partially_supported; else -> unsupported
```

This is **more lenient** than the LLM judge (it does not measure semantic
entailment). Both failures above measure ~50–55% overlap (the claim shares
topical tokens with the wrong chunk), so the deterministic proxy labels them
`partially_supported` instead of `unsupported`. The LLM judge (run earlier,
before the outage) correctly marked both `unsupported`.

Implication for reading the metrics: `support=0 unsupported` is an artifact of
the offline proxy. The trustworthy numbers are **coverage 100%** and
**traceability 130/130** (both deterministic) and, when credits return, the LLM
judge re-run (resumable via `scripts/evaluate_citations.py`) which previously
measured **109/130 supported, 10/130 partially, 2/130 unsupported**.

---

## Summary

| Failure | Stage | Pattern | Evidence available elsewhere? |
|---|---|---|---|
| H12 C1 → chunk 240 | Citations | relevant-but-not-supporting (extra adjacent chunk) | Yes — chunk 134 supports the claim |
| Q16 C2 → chunk 20 | Citations | relevant-but-not-supporting (list referenced, not enumerated) | Yes — chunk 139 supports the claim |

Both failures share one lesson: **a citation's job is to support the specific
claim, not just to be relevant to the question.** Retrieval relevance
(`relevant`) and citation support (`supported`) are deliberately measured as
separate dimensions in Phase 9.