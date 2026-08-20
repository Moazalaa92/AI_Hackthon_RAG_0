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
1. PATIENT_SPECIFIC detection starts with a small DETERMINISTIC personal-reference
   gate (zero external LLM calls). It covers first-person requests, age
   vignettes, singular individual references, and selected third-person clinical
   narratives without trying to become a policy engine.
2. An optional layer-2 LLM intent classifier can run only after layer 1 returns
   NORMAL. It is disabled by default for reproducible offline evaluation and
   falls back to layer 1 after two provider attempts.
3. INSUFFICIENT_EVIDENCE is INFERRED from the frozen Generation V2 refusal
   response. This is a signal, NOT independent proof that the evidence was
   objectively insufficient. H12 demonstrated over-abstention: the frozen model
   refused even when relevant evidence was present. We therefore keep explicit:
   `is_refusal_claim(answer)` means "the generation system produced a refusal",
   not "the evidence is objectively insufficient." No new LLM evidence judge is
   added; no lexical evidence rule (e.g. "chunk contains 'dose'") is used - such
   rules were shown to be unsafe.
"""

import json
import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    SAFETY_INTENT_LLM_ENABLED,
)
from src.pipeline import DEFAULT_STORE, Answer
from src.pipeline import query as pipeline_query
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

# Compact personal-reference markers. Deliberately small - this is a gate, not
# a policy engine. The deterministic layer is designed to match 0/94 frozen
# impersonal guideline questions.
_PATIENT_REFERENCE_RE = re.compile(
    r"\b(my|i|me|i'm|we|our|us)\b",
    re.IGNORECASE,
)
_AGE_VIGNETTE_RE = re.compile(
    r"\b(?:a|an|the)?\s*\d{1,3}\s*[- ]\s*year[- ]old\b",
    re.IGNORECASE,
)
_SINGULAR_REFERENCE_RE = re.compile(
    r"\b(?:this|the)\s+(?:patient|child|mother|father|son|daughter|"
    r"brother|sister)\b",
    re.IGNORECASE,
)
_THIRD_PERSON_PRONOUN_RE = re.compile(
    r"\b(?:he|she|they)\b"
    r"(?=[^.!?]{0,100}\b(?:has|have|had|is|are|was|were|takes?|taking|"
    r"wants?|needs?|reports?|experienced?|changed|stopped|started)\b)"
    r"(?=[^.!?]{0,100}\b(?:seizure|epilepsy|pregnan|medication|medicine|"
    r"valproate|dose|symptom|side effect|rash|diagnos|treatment|surgery|"
    r"scan|MRI|EEG)\b)[^.!?]*",
    re.IGNORECASE,
)
_THIRD_PERSON_NARRATIVE_RE = re.compile(
    r"\b(?:a|an|the)\s+(?:adult|child|patient|woman|man)\b"
    r"(?=[^.!?]{0,100}\b(?:has|have|had|is|are|was|were|takes?|taking|"
    r"wants?|needs?|reports?|experienced?|changed|stopped|started)\b)"
    r"(?=[^.!?]{0,100}\b(?:seizure|epilepsy|pregnan|medication|medicine|"
    r"valproate|dose|symptom|side effect|rash|diagnos|treatment|surgery|"
    r"scan|MRI|EEG)\b)[^.!?]*",
    re.IGNORECASE,
)

_INTENT_RUBRIC = """\
You classify whether a public user's question requests individualized medical
advice about a specific person or asks a general guideline question.

Return PATIENT_SPECIFIC for a personal decision, symptom, treatment, dose,
investigation, or safety question about the user or a specific individual.
Return NORMAL for an impersonal question asking what the guideline recommends,
even when it discusses patients, medicines, pregnancy, driving, or surgery in
general.

Respond with exactly one JSON object and no surrounding text:
{"classification": "NORMAL" or "PATIENT_SPECIFIC", "reason": "one short sentence"}
"""


def classify_intent(question: str) -> SafetyResult:
    """Pre-retrieval intent gate.

    Layer 1 deterministically detects first-person references, age vignettes,
    singular individual references, and selected third-person clinical
    narratives. If layer 1 returns NORMAL and the opt-in environment flag is
    enabled, layer 2 asks the configured LLM for an intent classification.
    Provider failures retry once, then preserve the layer-1 result.
    """
    q = str(question or "").strip()
    if not q:
        return SafetyResult(
            classification=NORMAL,
            reason="layer 1 deterministic gate: empty question (handled by API validation)",
            safe_to_answer=True,
        )
    matchers = (
        ("first-person/personal reference", _PATIENT_REFERENCE_RE),
        ("age vignette", _AGE_VIGNETTE_RE),
        ("singular individual reference", _SINGULAR_REFERENCE_RE),
        ("third-person pronoun clinical narrative", _THIRD_PERSON_PRONOUN_RE),
        ("third-person clinical narrative", _THIRD_PERSON_NARRATIVE_RE),
    )
    for label, matcher in matchers:
        match = matcher.search(q)
        if match:
            return SafetyResult(
                classification=PATIENT_SPECIFIC,
                reason=(
                    f"layer 1 deterministic gate: {label} detected "
                    f"({match.group(0)!r})"
                ),
                safe_to_answer=False,
            )

    layer_one = SafetyResult(
        classification=NORMAL,
        reason="layer 1 deterministic gate: no patient-specific reference detected",
        safe_to_answer=True,
    )
    if not SAFETY_INTENT_LLM_ENABLED:
        return layer_one
    return _classify_intent_with_llm(q, layer_one)


def _build_intent_client():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        max_tokens=200,
    )


def _parse_intent_response(raw_text):
    text = raw_text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in response")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise TypeError("JSON response is not an object")
    classification = data.get("classification")
    reason = data.get("reason")
    if classification not in (NORMAL, PATIENT_SPECIFIC):
        raise ValueError(
            f"classification must be {NORMAL!r} or {PATIENT_SPECIFIC!r}"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    return classification, reason.strip()


def _classify_intent_with_llm(question, layer_one):
    """Run the optional layer-2 classifier, preserving layer-1 on failure."""
    last_error = None
    for attempt in (1, 2):
        prompt = f"QUESTION:\n{question}\n\nReturn only the JSON object."
        if attempt == 2 and last_error:
            prompt += f"\n\nPrevious attempt failed: {last_error}."
        try:
            client = _build_intent_client()
            response = client.invoke(
                [SystemMessage(content=_INTENT_RUBRIC), HumanMessage(content=prompt)]
            )
            raw = response.content
            if isinstance(raw, list):
                raw = "".join(part.get("text", "") for part in raw)
            if not raw or not raw.strip():
                raise ValueError("empty response")
            classification, reason = _parse_intent_response(raw)
            return SafetyResult(
                classification=classification,
                reason=f"layer 2 LLM intent classifier: {reason}",
                safe_to_answer=classification == NORMAL,
            )
        except Exception as exc:  # noqa: BLE001 - provider errors must fail open to layer 1
            last_error = f"{type(exc).__name__}: {exc}"
    return SafetyResult(
        classification=layer_one.classification,
        reason=(
            "layer 2 LLM intent classifier failed after 2 attempts; "
            f"keeping layer 1 result ({layer_one.classification}): {last_error}"
        ),
        safe_to_answer=layer_one.safe_to_answer,
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