"""Safety / intent layer (Phase 11): a thin, explicit guardrail over the frozen RAG.

This module ONLY classifies and refuses; it never re-implements retrieval,
generation, or citation logic. All RAG work is delegated to the frozen
src.pipeline.query(). The safety layer sits ABOVE the pipeline:

    FastAPI /ask
        -> classify_intent(question)          (deterministic, pre-retrieval)
              |-- PATIENT_SPECIFIC -> safe refusal (pipeline NOT called)
              `-- NORMAL -> pipeline.query() -> frozen RAG flow -> Answer
                  -> classify_evidence(answer, claims)  (post-hoc, refusal signal)
                  -> SafetyResult

Three-way classification (explicit contract):

    NORMAL               safe_to_answer = True
    PATIENT_SPECIFIC     safe_to_answer = False
    INSUFFICIENT_EVIDENCE safe_to_answer = False

`safe_to_answer` means exactly: "allowed to proceed under our current safety
policy." It does NOT mean clinically safe, medically validated, or guaranteed
correct.

Design decisions (approved):
1. PATIENT_SPECIFIC detection is a DETERMINISTIC first-person / personal-reference
   rule gate (zero external LLM calls). Known limitation: third-person scenario
   formulations (e.g. "A 7-year-old has recurrent absence seizures...") are NOT
   reliably caught - this is measured, not hidden.
2. INSUFFICIENT_EVIDENCE is INFERRED from the frozen Generation V2 refusal
   response. This is a signal, NOT independent proof that the evidence was
   objectively insufficient. H12 demonstrated over-abstention: the frozen model
   refused even when relevant evidence was present. We therefore keep explicit:
   `is_refusal_claim(answer)` means "the generation system produced a refusal",
   not "the evidence is objectively insufficient." No new LLM evidence judge is
   added; no lexical evidence rule (e.g. "chunk contains 'dose'") is used - such
   rules were shown to be unsafe.
"""

import re
from dataclasses import dataclass

from src.pipeline import DEFAULT_STORE, Answer, query as pipeline_query
from src.sources import is_refusal_claim, split_claims

# --- classification contract -------------------------------------------------

NORMAL = "NORMAL"
PATIENT_SPECIFIC = "PATIENT_SPECIFIC"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

PATIENT_SPECIFIC_REFUSAL = (
    "I can answer general questions from the clinical guideline, but I can't "
    "provide individualized medical advice. Please consult a qualified "
    "healthcare professional for a decision about a specific patient."
)


@dataclass
class SafetyResult:
    """Result of the safety/intent check for one question.

    `safe_to_answer` means "allowed to proceed under the current safety policy."
    It does NOT mean clinically safe, medically validated, or guaranteed correct.
    """

    classification: str
    reason: str
    safe_to_answer: bool


# --- deterministic patient-specific gate -------------------------------------

# Compact first-person / personal-reference markers. Deliberately small - this
# is a gate, not a policy engine. Matches 0/94 frozen guideline questions (the
# guideline corpus is consistently impersonal).
_PATIENT_REFERENCE_RE = re.compile(
    r"\b(my|i|me|i'm|we|our|us)\b",
    re.IGNORECASE,
)


def classify_intent(question: str) -> SafetyResult:
    """Pre-retrieval intent gate.

    Detects first-person / personal-reference formulations (e.g. "my child",
    "should I", "for me") as PATIENT_SPECIFIC. Everything else is NORMAL.
    Pure regex: cannot fail at runtime, no external API, fail-closed by design.

    Known limitation: third-person scenario formulations ("A 7-year-old has
    recurrent absence seizures. Which medication should be started?") have no
    first-person marker and are classified NORMAL. This is measured in the
    safety evaluation, not hidden.
    """
    q = str(question or "").strip()
    if not q:
        return SafetyResult(
            classification=NORMAL,
            reason="empty question (handled by API validation)",
            safe_to_answer=True,
        )
    match = _PATIENT_REFERENCE_RE.search(q)
    if match:
        return SafetyResult(
            classification=PATIENT_SPECIFIC,
            reason=(
                f"first-person/personal reference detected "
                f"({match.group(0)!r}) in the question"
            ),
            safe_to_answer=False,
        )
    return SafetyResult(
        classification=NORMAL,
        reason="no first-person/personal reference detected",
        safe_to_answer=True,
    )


# --- insufficient-evidence interpretation (refusal signal) -------------------

# The frozen citation layer's REFUSAL_PHRASES (src.sources) cover the canonical
# refusal phrasings, but the frozen list is intentionally narrow (it protects
# citation-building for refusal claims). The safety layer is a SEPARATE concern
# and adds a small, explicit, documented superset so that genuine evidence
# refusals such as "The context does not contain the recommended starting
# dose..." are detected. This is new safety-layer code, NOT a change to the
# frozen sources.py. The check is a signal, never proof of objective
# insufficiency (H12 over-abstention can still occur - measured separately).
_REFUSAL_MARKERS = (
    "not found",
    "does not contain",
    "does not provide",
    "does not include",
    "does not specify",
    "is not provided",
    "is not given",
    "not available",
    "not provided",
    "cannot provide",
    "cannot answer",
    "unable to answer",
)


def _claim_texts(claims):
    for claim in claims or []:
        text = claim.get("claim_text") if isinstance(claim, dict) else None
        if text:
            yield text


def _is_refusal_text(text: str) -> bool:
    """True if `text` is an evidence-refusal statement (marker match)."""
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _is_refusal_response(answer, claims) -> bool:
    """True if the frozen generation produced an evidence-refusal response.

    Explicit interpretation: this means "the generation system produced a
    refusal", NOT "the evidence was objectively insufficient." Over-abstention
    (e.g. H12) can occur; the safety evaluation measures false refusals.

    A response is a refusal response only if EVERY claim is a refusal (mixed
    answers that do answer are not refusals). Each claim is checked with both
    the frozen citation-layer detector (src.sources.is_refusal_claim) and the
    safety-layer marker set above.
    """
    claim_texts = list(_claim_texts(claims))
    if claim_texts:
        return all(
            is_refusal_claim(text) or _is_refusal_text(text)
            for text in claim_texts
        )
    if answer:
        parts = split_claims(str(answer))
        if parts:
            return all(
                is_refusal_claim(p) or _is_refusal_text(p)
                for p in parts
            )
    return False


def classify_evidence(answer: str, claims: list) -> SafetyResult:
    """Post-hoc interpretation of the frozen pipeline result.

    If the generated answer consists of refusal claims, classify as
    INSUFFICIENT_EVIDENCE and refuse. Otherwise NORMAL.

    The generated answer is preserved verbatim - never replaced.
    """
    if _is_refusal_response(answer, claims):
        return SafetyResult(
            classification=INSUFFICIENT_EVIDENCE,
            reason=(
                "frozen Generation V2 produced an evidence-refusal response "
                "(signal, not independent proof of insufficient evidence)"
            ),
            safe_to_answer=False,
        )
    return SafetyResult(
        classification=NORMAL,
        reason="generated answer is not an evidence-refusal response",
        safe_to_answer=True,
    )


# --- orchestrator ------------------------------------------------------------

def _patient_specific_answer(question: str) -> Answer:
    """A safe refusal Answer - never calls the RAG pipeline."""
    return Answer(
        question=question,
        answer=PATIENT_SPECIFIC_REFUSAL,
        claims=[],
        retrieved_chunks=[],
        citations_valid=False,
        validation_errors=[],
    )


def query(
    question: str,
    *,
    persist_dir: str = DEFAULT_STORE,
    top_k: int = 10,
    max_tokens: int = 600,
    selector=None,
):
    """Run the safety-aware flow and return (Answer, SafetyResult).

    - PATIENT_SPECIFIC: returns a safe refusal Answer (pipeline NOT executed).
    - Otherwise: runs the frozen pipeline unchanged, then interprets the result.
    """
    intent = classify_intent(question)
    if not intent.safe_to_answer:
        return _patient_specific_answer(question), intent

    result = pipeline_query(
        question,
        persist_dir=persist_dir,
        top_k=top_k,
        max_tokens=max_tokens,
        selector=selector,
    )
    evidence = classify_evidence(result.answer, result.claims)
    return result, evidence