"""Citation/source layer (Phase 9): answer claims -> exact retrieved evidence.

This module is the DETERMINISTIC core of the citation pipeline. It never
calls an LLM itself; the claim->chunk selector is injected by the caller so
this module stays reproducible and testable. The rule enforced here:

    a citation may ONLY point to a chunk that was ACTUALLY retrieved for the
    question (present in the frozen Top-10), and every field of a citation
    (chunk_id, page, page_label, source, supporting_text, ...) is COPIED from
    the retrieved chunk record - never invented.

Pipeline:
    answer text
      -> split_claims()                       (deterministic sentence split)
      -> pick_candidates()                    (deterministic lexical pre-filter)
      -> selector(question, claim, candidates)   (LLM chooses among candidates)
      -> make_citation()                      (build record from chunk metadata)
      -> validate_citations()                 (deterministic safety checks)

The selector is constrained: it can only choose from the pre-filtered
candidates, all of which are retrieved chunks for this question. Even a bad
selector cannot fabricate a page, a source, or a chunk that was not retrieved.
"""

import re

ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "p.",
    "pp.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "etc.",
    "Fig.",
    "No.",
    "Nos.",
    "vs.",
    "approx.",
    "St.",
    "ref.",
    "cf.",
)

REFUSAL_PHRASES = (
    "not found in the provided",
    "does not contain enough information",
    "answer was not found",
    "could not be found",
    "not available in the provided",
    "the provided context does not contain",
    "does not specify",
    "is not provided in the context",
    "not provided in the context",
    "is not given in the context",
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

_SENTINEL = "\u0001"


def _stash_patterns(text):
    """Protect decimals/recommendation numbers and abbreviations before splitting.

    Returns (text_with_sentinels, placeholders) where placeholders[n] is the
    original substring replaced by '{sentinel}n{sentinel}'.
    """
    placeholders = []

    def _stash(match):
        placeholders.append(match.group(0))
        return f"{_SENTINEL}{len(placeholders) - 1}{_SENTINEL}"

    text = re.sub(r"\b\d+\.\d+(?:\.\d+)*\b", _stash, text)
    abbr_pattern = r"(?i)\b(?:" + "|".join(re.escape(a) for a in ABBREVIATIONS) + r")\b"
    text = re.sub(abbr_pattern, _stash, text)
    return text, placeholders


def _restore(text, placeholders):
    for i, original in enumerate(placeholders):
        text = text.replace(f"{_SENTINEL}{i}{_SENTINEL}", original)
    return text


def split_claims(answer):
    """Deterministically split an answer into sentence-level claims.

    Splits on '.', '!' or '?' followed by whitespace and a capital letter,
    digit, quote or opening parenthesis, and on newline bullet items
    ("- ", "* ", "• "). Decimal/recommendation numbers (e.g. "4.1.1", "1.5")
    and common abbreviations are protected first so they never trigger a
    split. Returns a list of non-empty cleaned sentences.
    """
    if not answer or not str(answer).strip():
        return []
    text = str(answer).strip()
    protected, placeholders = _stash_patterns(text)
    protected = re.sub(r"\n\s*[-*•]\s+", ". ", protected)
    protected = re.sub(r"\.{2,}", ".", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", protected)
    claims = []
    for part in parts:
        cleaned = _restore(part, placeholders)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            continue
        if cleaned.rstrip(".").endswith(":"):
            continue
        claims.append(cleaned)
    return claims


def _tokens(text):
    return set(_TOKEN_RE.findall(text.lower()))


def token_overlap(claim, chunk_text):
    """Fraction of claim tokens that also appear in the chunk text.

    Deterministic lexical pre-filter only. Used to rank candidate chunks before
    the selector LLM runs; never the final word on support.
    """
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return 0.0
    chunk_tokens = _tokens(chunk_text or "")
    if not chunk_tokens:
        return 0.0
    return len(claim_tokens & chunk_tokens) / len(claim_tokens)


def pick_candidates(claim, chunks, max_candidates=3, min_overlap=0.05):
    """Deterministically pre-filter retrieved chunks for a claim.

    Ranks chunks by token overlap with the claim and returns the top
    `max_candidates` with overlap >= `min_overlap`. If nothing meets the
    threshold but at least one chunk shares a token, the single best chunk is
    returned so the selector still has a chance. Only chunks already retrieved
    for the question can ever be returned.
    """
    scored = []
    for chunk in chunks:
        overlap = token_overlap(claim, chunk.get("chunk_text") or "")
        scored.append((overlap, chunk))
    scored.sort(key=lambda item: -item[0])
    candidates = [chunk for overlap, chunk in scored if overlap >= min_overlap]
    if not candidates and scored and scored[0][0] > 0:
        candidates = [scored[0][1]]
    return candidates[:max_candidates]


def is_refusal_claim(claim):
    """True if the claim is a refusal / 'not found' statement.

    Refusal claims never get citations: there is no supporting evidence for
    "the answer was not found". This is a deterministic guard so deliberate
    negatives (Q20/H26/H27) and over-abstentions do not accumulate bogus
    citations.
    """
    lowered = claim.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def make_citation(question_id, claim_id, citation_index, chunk):
    """Build a citation record from a retrieved chunk record.

    Every field is copied from the chunk record - nothing is invented. The
    `supporting_text` is the EXACT chunk text stored in the frozen retrieval
    record, so the citation is traceable to the precise source text.
    """
    return {
        "citation_id": f"{question_id}-{claim_id}-C{citation_index}",
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "source": chunk.get("source"),
        "page": chunk.get("page"),
        "page_label": chunk.get("page_label"),
        "reranker_rank": chunk.get("reranker_rank"),
        "reranker_score": chunk.get("reranker_score"),
        "rank": chunk.get("rank"),
        "fusion_score": chunk.get("fusion_score"),
        "retrieved_by": chunk.get("retrieved_by"),
        "relevant": chunk.get("relevant"),
        "supporting_text": chunk.get("chunk_text"),
    }


def build_citations(question_id, question, answer, chunks, selector=None):
    """Turn an answer + its retrieved chunks into claim-level citations.

    Args:
        question_id: question identifier (used in citation_id).
        question: the question text.
        answer: the generated answer text (plain string).
        chunks: the frozen Top-10 chunk records for this question.
        selector: callable(question, claim, candidates) -> list[chunk].
            Chooses which pre-filtered candidate chunks support the claim.
            If None, all candidates are cited (deterministic default).

    Returns:
        List of claim dicts:
            {"claim_id": "C1", "claim_text": "...", "citations": [citation, ...]}
        Refusal claims always get an empty citation list.
    """
    claims_out = []
    for index, claim_text in enumerate(split_claims(answer), start=1):
        claim_id = f"C{index}"
        citations = []
        if not is_refusal_claim(claim_text):
            candidates = pick_candidates(claim_text, chunks)
            if candidates:
                if selector is None:
                    selected = candidates
                else:
                    selected = selector(question, claim_text, candidates)
                    if selected is None:
                        selected = []
                for c_index, chunk in enumerate(selected, start=1):
                    citations.append(
                        make_citation(question_id, claim_id, c_index, chunk)
                    )
        claims_out.append({"claim_id": claim_id, "claim_text": claim_text, "citations": citations})
    return claims_out


def validate_citations(claims, chunks):
    """Deterministically validate every citation against the retrieved chunks.

    Checks, per citation:
      - cited chunk_id is present in the retrieved set for this question
      - supporting_text equals the chunk's actual text (no invented text)
      - page/page_label/source match the chunk's metadata (no invented pages)

    Returns (valid: bool, errors: list[str]).
    """
    by_id = {chunk.get("chunk_id"): chunk for chunk in chunks}
    retrieved_ids = set(by_id)
    errors = []
    for claim in claims:
        claim_id = claim["claim_id"]
        for citation in claim["citations"]:
            cid = citation.get("chunk_id")
            if cid not in retrieved_ids:
                errors.append(
                    f"{claim_id}: cites chunk {cid!r} which was not retrieved for "
                    f"this question"
                )
                continue
            chunk = by_id[cid]
            if citation.get("supporting_text") != chunk.get("chunk_text"):
                errors.append(
                    f"{claim_id}: supporting_text does not match chunk {cid!r} text"
                )
            if citation.get("page") != chunk.get("page"):
                errors.append(f"{claim_id}: page mismatch for chunk {cid!r}")
            if citation.get("page_label") != chunk.get("page_label"):
                errors.append(f"{claim_id}: page_label mismatch for chunk {cid!r}")
            if citation.get("source") != chunk.get("source"):
                errors.append(f"{claim_id}: source mismatch for chunk {cid!r}")
    return len(errors) == 0, errors
