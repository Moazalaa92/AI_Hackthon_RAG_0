"""LLM judges for the citation/source layer (Phase 9).

Two distinct roles, following the same client / retry / JSON-parse / validation
pattern as src/judge.py and src/generation_judge.py:

  A. select_supporting_chunks (build-time mapper)
     Receives ONLY: question, claim, and the pre-filtered CANDIDATE chunks that
     were retrieved for the question. Picks which candidates actually support
     the claim. This is a MAPPER, not ground truth: it can only choose from the
     retrieved candidates it is given, and the final citation records are still
     validated deterministically by src/sources.validate_citations.

  B. judge_citation_support (evaluation judge)
     Receives ONLY: question, claim, and the cited chunk's supporting_text.
     Labels each citation supported / partially_supported / unsupported.
     Explicitly an EVALUATOR, not the source of truth.

Both are LLM-as-judge: their verdicts are engineering measurements, not proof.
"""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

SELECTOR_RUBRIC = """\
You are an expert annotator selecting the retrieved chunks that support a \
single claim made by a RAG answer.

You will receive:
- QUESTION: the question the system answered
- CLAIM: one factual statement from the system's answer
- CANDIDATE CHUNKS: numbered chunks that were retrieved for this question

Select ALL candidate chunks whose text directly supports the claim. A chunk \
supports the claim if its text contains the information, or directly implies \
it. A chunk that merely discusses the same general topic (for example, a \
nearby but different recommendation) does NOT support the claim.

Important rules:
1. Only select chunks that are actually present in the candidate list.
2. Do not select a chunk just because it is related or topically similar — \
the claim must be supported by the chunk's text.
3. Select zero chunks if no candidate supports the claim.

You MUST respond with a single JSON object and nothing else, in exactly this \
format:
{"chunk_numbers": [1, 3], "reason": "one short sentence explaining the decision"}
"""

SUPPORT_RUBRIC = """\
You are an expert annotator evaluating whether a cited chunk actually supports \
a claim made by a RAG answer. This evaluates citation CORRECTNESS (does the \
evidence back the claim), not whether the claim is true in the real world.

You will receive:
- QUESTION: the question the system answered
- CLAIM: one factual statement from the system's answer
- CITED CHUNK TEXT: the exact supporting text the citation points to

Labels:
- "supported": the cited chunk text directly contains, or directly implies, \
the claim.
- "partially_supported": the cited chunk text supports part of the claim but \
not all of it (e.g. a multi-part claim where the chunk covers only some parts).
- "unsupported": the cited chunk text does not contain or imply the claim. \
A chunk that is on the same general topic but does not state the claim is \
"unsupported".

Important rules:
1. Judge only against the CITED CHUNK TEXT provided. Do not use outside \
knowledge or the retrieval context.
2. A chunk being relevant to the question is NOT the same as supporting the \
exact claim.

You MUST respond with a single JSON object and nothing else, in exactly this \
format:
{"label": "supported" or "partially_supported" or "unsupported", "reason": \
"one short sentence explaining the decision"}
"""


def _build_client(max_tokens=800):
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        max_tokens=max_tokens,
    )


def _extract_json(raw_text):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in response: {raw_text!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e


class SourceJudgeError(Exception):
    """Raised when a citation judge fails after retries."""

    def __init__(self, category, message):
        super().__init__(f"[{category}] {message}")
        self.category = category


def _invoke(rubric, prompt, max_tokens=800):
    """Run one judge call with retries; returns parsed JSON dict."""
    client = _build_client(max_tokens=max_tokens)
    last_error = None
    for attempt in (1, 2, 3):
        text = prompt
        if attempt > 1 and last_error is not None:
            text += (
                f"\n\nYour previous response was invalid: {last_error}. "
                f"Return exactly one valid JSON object."
            )
        try:
            response = client.invoke(
                [SystemMessage(content=rubric), HumanMessage(content=text)]
            )
            raw = response.content
            if isinstance(raw, list):
                raw = "".join(part.get("text", "") for part in raw)
            if not raw or not raw.strip():
                raise SourceJudgeError("empty_response", "model returned an empty response")
            return _extract_json(raw)
        except (SourceJudgeError, ValueError) as e:
            if attempt == 3:
                raise SourceJudgeError("malformed_response", str(e)) from e
            last_error = str(e)
        except Exception as e:
            if attempt == 3:
                raise SourceJudgeError("api_error", f"{type(e).__name__}: {e}") from e
            last_error = f"{type(e).__name__}: {e}"
    raise SourceJudgeError("other", "unreachable")  # pragma: no cover


def select_supporting_chunks(question, claim, candidates):
    """Select the candidate chunks that support `claim`.

    Args:
        question: the question text.
        claim: the claim text.
        candidates: list of chunk records (from src.sources.pick_candidates).

    Returns:
        list of candidate chunk records that support the claim (subset of
        `candidates`; possibly empty).

    Raises:
        SourceJudgeError: if the model cannot be parsed after retries.
    """
    numbered = []
    for i, chunk in enumerate(candidates, start=1):
        text = chunk.get("chunk_text") or ""
        numbered.append(
            f"[{i}] (chunk_id={chunk.get('chunk_id')}, "
            f"page_label={chunk.get('page_label')}, rank={chunk.get('reranker_rank')})\n{text}"
        )
    data = _invoke(
        SELECTOR_RUBRIC,
        f"QUESTION:\n{question}\n\n"
        f"CLAIM:\n{claim}\n\n"
        f"CANDIDATE CHUNKS:\n" + "\n\n".join(numbered) + "\n\n"
        "Keep the reason to one short sentence. Return only the JSON object.",
        max_tokens=2000,
    )
    if not isinstance(data, dict):
        raise SourceJudgeError("malformed_response", f"JSON is not an object: {data!r}")
    numbers = data.get("chunk_numbers")
    reason = data.get("reason")
    if not isinstance(numbers, list):
        raise SourceJudgeError(
            "malformed_response", f"'chunk_numbers' must be a list, got {numbers!r}"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise SourceJudgeError(
            "malformed_response", f"'reason' must be a non-empty string, got {reason!r}"
        )
    selected = set()
    for n in numbers:
        if not isinstance(n, int) or n < 1 or n > len(candidates):
            raise SourceJudgeError(
                "malformed_response",
                f"chunk number {n!r} outside candidate range 1..{len(candidates)}",
            )
        selected.add(n)
    return [candidates[i - 1] for i in sorted(selected)]


def _support_tokens(text):
    """Lowercased alphanumeric tokens for the deterministic fallback judge."""
    return set(_TOKEN_RE.findall((text or "").lower()))


def judge_citation_support_deterministic(question, claim, supporting_text):
    """Deterministic offline proxy for citation support (no external API).

    NOT a replacement for the LLM evaluator: it is a transparent lexical
    heuristic used only when the LLM API is unavailable, so the pipeline can
    still produce reproducible verdicts. Verdicts are tagged with
    judge="deterministic" and reported separately from the LLM judge.

    Logic: fraction of the claim's tokens that appear in the cited chunk text.
        >= 0.66  -> supported
        >= 0.33  -> partially_supported
        otherwise -> unsupported

    Returns {"label": ..., "reason": ..., "judge": "deterministic"}.
    """
    claim_tokens = _support_tokens(claim)
    chunk_tokens = _support_tokens(supporting_text)
    if not claim_tokens:
        return {
            "label": "unsupported",
            "reason": "claim has no lexical tokens",
            "judge": "deterministic",
        }
    overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens)
    if overlap >= 0.66:
        label = "supported"
    elif overlap >= 0.33:
        label = "partially_supported"
    else:
        label = "unsupported"
    return {
        "label": label,
        "reason": f"deterministic token overlap {overlap:.0%} of claim in cited chunk",
        "judge": "deterministic",
    }


def judge_citation_support(question, claim, supporting_text):
    """Judge whether the cited chunk text supports the claim (evaluator only).

    Receives ONLY question + claim + cited supporting text. No retrieval
    context, no reference answer, no other chunks.

    Primary path is the LLM evaluator. If the LLM call fails (e.g. the API is
    unreachable or credits are exhausted -> SourceJudgeError category
    'api_error'), it falls back to the deterministic lexical proxy so the
    evaluation can still complete. Every verdict is tagged with 'judge':
    'llm' or 'deterministic'.

    Returns {"label": supported|partially_supported|unsupported,
             "reason": ..., "judge": "llm"|"deterministic"}
    """
    try:
        data = _invoke(
            SUPPORT_RUBRIC,
            f"QUESTION:\n{question}\n\n"
            f"CLAIM:\n{claim}\n\n"
            f"CITED CHUNK TEXT:\n{supporting_text}\n\n"
            "Return only the JSON object.",
        )
    except SourceJudgeError as e:
        if e.category == "api_error":
            result = judge_citation_support_deterministic(question, claim, supporting_text)
            result["llm_error"] = str(e)
            return result
        raise
    if not isinstance(data, dict):
        raise SourceJudgeError("malformed_response", f"JSON is not an object: {data!r}")
    label = data.get("label")
    reason = data.get("reason")
    allowed = ("supported", "partially_supported", "unsupported")
    if label not in allowed:
        raise SourceJudgeError(
            "malformed_response", f"'label' must be one of {allowed}, got {label!r}"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise SourceJudgeError(
            "malformed_response", f"'reason' must be a non-empty string, got {reason!r}"
        )
    return {"label": label, "reason": reason.strip(), "judge": "llm"}
