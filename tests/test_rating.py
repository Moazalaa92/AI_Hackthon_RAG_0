from src.pipeline import Answer
from src.rating import T_MARGIN, T_TOP1, rate_answer
from src.safety import INSUFFICIENT_EVIDENCE, NORMAL, PATIENT_SPECIFIC, SafetyResult


def _safety(classification=NORMAL):
    return SafetyResult(classification, "test", classification == NORMAL)


def _answer(
    *,
    citations_valid=True,
    claims=None,
    scores=(3.0, 2.0, 1.0),
    retrieved_by="dense+bm25",
):
    chunks = [
        {
            "page": 4,
            "reranker_score": score,
            "retrieved_by": retrieved_by,
        }
        for score in scores
    ]
    return Answer(
        question="q",
        answer="Grounded answer.",
        claims=claims if claims is not None else [{"citations": [{"page": 4}]}],
        retrieved_chunks=chunks,
        citations_valid=citations_valid,
    )


def test_refused_for_non_normal_safety():
    assert rate_answer(_answer(), _safety(PATIENT_SPECIFIC)).band == "refused"
    assert rate_answer(_answer(), _safety(INSUFFICIENT_EVIDENCE)).band == "refused"


def test_low_for_invalid_citations():
    assert rate_answer(_answer(citations_valid=False), _safety()).band == "low"
    assert rate_answer(_answer(citations_valid=None), _safety()).band == "low"


def test_low_for_empty_claims_and_low_coverage():
    assert rate_answer(_answer(claims=[]), _safety()).band == "low"
    answer = _answer(
        claims=[
            {"citations": []},
            {"citations": []},
            {"citations": [{"page": 4}]},
        ]
    )
    assert rate_answer(answer, _safety()).band == "low"


def test_high_requires_thresholds_and_complete_citations():
    result = rate_answer(_answer(), _safety())
    assert result.band == "high"
    assert result.signals["top1_reranker_score"] >= T_TOP1
    assert result.signals["top1_top3_margin"] >= T_MARGIN


def test_high_boundary_and_dual_retrieval():
    answer = _answer(scores=(T_TOP1, 0.0, T_TOP1 - T_MARGIN))
    assert rate_answer(answer, _safety()).band == "high"
    answer = _answer(
        scores=(T_TOP1, T_TOP1, T_TOP1),
        retrieved_by="dense+bm25",
    )
    assert rate_answer(answer, _safety()).band == "high"


def test_medium_when_valid_but_not_high():
    answer = _answer(scores=(T_TOP1 - 0.01, 0.0, T_TOP1 - 0.01))
    assert rate_answer(answer, _safety()).band == "medium"


def test_refusal_answer_has_no_support_claim():
    answer = _answer()
    answer = Answer(
        question=answer.question,
        answer="The answer was not found in the provided documents.",
        claims=[],
        retrieved_chunks=answer.retrieved_chunks,
        citations_valid=True,
    )
    assert rate_answer(answer, _safety()).band == "refused"
