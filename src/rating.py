"""Deterministic evidence-support bands for public answers.

This module is deliberately offline and rule-based. It does not assess
clinical correctness; it summarizes observable retrieval, citation, and safety
signals into a conservative support band.
"""

from dataclasses import dataclass

from src.pipeline import Answer
from src.safety import NORMAL, SafetyResult
from src.sources import is_refusal_claim, split_claims

# Provisional calibration values. Phase B may override these through the
# function arguments without changing the rule table.
T_TOP1 = 1.0
T_MARGIN = 0.5


@dataclass(frozen=True)
class RatingResult:
    """Support band, machine-readable signals, and short explanations."""

    band: str
    signals: dict
    reasons: list[str]


def _is_refusal_answer(answer: Answer) -> bool:
    claims = answer.claims or []
    if claims:
        claim_texts = [
            claim.get("claim_text", "")
            for claim in claims
            if isinstance(claim, dict)
        ]
        claim_texts = [text for text in claim_texts if text]
        return bool(claim_texts) and all(is_refusal_claim(text) for text in claim_texts)
    return bool(split_claims(answer.answer)) and all(
        is_refusal_claim(text) for text in split_claims(answer.answer)
    )


def _signals(answer: Answer, safety: SafetyResult) -> dict:
    chunks = answer.retrieved_chunks or []
    top1 = chunks[0] if chunks else {}
    raw_top1_score = top1.get("reranker_score")
    raw_top3_score = chunks[2].get("reranker_score") if len(chunks) >= 3 else None
    top1_score = float(raw_top1_score) if raw_top1_score is not None else None
    top3_score = float(raw_top3_score) if raw_top3_score is not None else None
    margin = (
        float(top1_score) - float(top3_score)
        if top1_score is not None and top3_score is not None
        else None
    )
    top1_page = top1.get("page")
    page_agreement = (
        sum(chunk.get("page") == top1_page for chunk in chunks[:5])
        if top1_page is not None
        else 0
    )
    retrieved_by = str(top1.get("retrieved_by") or "").lower()
    dual_retrieval = "dense" in retrieved_by and "bm25" in retrieved_by
    claims = answer.claims or []
    cited_claim_count = sum(bool(claim.get("citations")) for claim in claims)
    claim_count = len(claims)
    coverage = cited_claim_count / claim_count if claim_count else 0.0
    return {
        "top1_reranker_score": top1_score,
        "top1_top3_margin": margin,
        "page_agreement_top5": page_agreement,
        "dual_retrieval": dual_retrieval,
        "claim_count": claim_count,
        "cited_claim_count": cited_claim_count,
        "claim_citation_coverage": coverage,
        "citations_valid": answer.citations_valid,
        "safety_classification": safety.classification,
    }


def rate_answer(
    answer: Answer,
    safety: SafetyResult,
    *,
    top1_threshold: float = T_TOP1,
    margin_threshold: float = T_MARGIN,
) -> RatingResult:
    """Assign the contract band to an answer and its safety result."""
    signals = _signals(answer, safety)
    if safety.classification != NORMAL or _is_refusal_answer(answer):
        return RatingResult(
            band="refused",
            signals=signals,
            reasons=["The request or answer was refused; no support claim is made."],
        )
    if (
        answer.citations_valid is not True
        or not answer.claims
        or signals["claim_citation_coverage"] < 0.5
    ):
        return RatingResult(
            band="low",
            signals=signals,
            reasons=[
                "Citation validation or claim coverage is insufficient for a stronger band."
            ],
        )
    high_score = (
        signals["top1_reranker_score"] is not None
        and signals["top1_reranker_score"] >= top1_threshold
    )
    high_margin = (
        signals["top1_top3_margin"] is not None
        and signals["top1_top3_margin"] >= margin_threshold
    )
    if (
        signals["claim_citation_coverage"] == 1.0
        and high_score
        and (high_margin or signals["dual_retrieval"])
    ):
        return RatingResult(
            band="high",
            signals=signals,
            reasons=["Citations are complete and the top evidence signals are strong."],
        )
    return RatingResult(
        band="medium",
        signals=signals,
        reasons=["Citations are valid, but the stronger high-band thresholds are not all met."],
    )
