"""LLM relevance judge for retrieval evaluation (Phase 12.5).

The judge receives ONLY:
    - the question
    - the retrieved chunk text
    - the relevance rubric

It evaluates retrieval relevance — "does this chunk contribute toward answering
the SPECIFIC question?" — NOT clinical correctness against an external source.
No expected_pages, no PDF, no reference answers are provided.
"""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

RUBRIC = """\
You are an expert annotator judging whether a retrieved text chunk is relevant \
to a user's question. This is for evaluating a RETRIEVAL system, not for \
generating an answer.

Definitions:
- "relevant": the chunk contains information that contributes toward answering \
the SPECIFIC question being asked (the exact recommendation, definition, \
criterion, etc. that the question requests).
- "not_relevant": the chunk merely discusses the same general topic or a \
related concept but does not contribute toward answering the specific question.

Important rules:
1. Do NOT reduce relevance to simple keyword or topic similarity. A chunk about \
the same disease or the same treatment in a different context (for example, a \
first-line add-on treatment when the question asks for first-line monotherapy) \
is NOT relevant.
2. Judge only whether the chunk helps answer THIS question. Partial but \
directly on-point information counts as relevant.
3. Do not assume knowledge of any external guideline. Judge only from the \
chunk text provided, and from the question.
4. If the chunk has no usable information about what the question asks, the \
answer is "not_relevant".

You MUST respond with a single JSON object and nothing else, in exactly this \
format:
{"relevant": "relevant" or "not_relevant", "reason": "one short sentence \
explaining the decision"}
"""


def _build_client():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        max_tokens=400,
    )


def _parse_judgment(raw_text):
    """Extract and validate a judgment from a raw model response.

    Returns a dict {"relevant": str, "reason": str}. Raises ValueError on any
    malformed or out-of-spec response.
    """
    text = raw_text.strip()
    # Extract the JSON object, tolerating a little surrounding text.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in response: {raw_text!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"JSON is not an object: {data!r}")

    relevant = data.get("relevant")
    reason = data.get("reason")

    if relevant not in ("relevant", "not_relevant"):
        raise ValueError(f"'relevant' must be 'relevant' or 'not_relevant', got {relevant!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"'reason' must be a non-empty string, got {reason!r}")

    return {"relevant": relevant, "reason": reason.strip()}


class JudgeError(Exception):
    """Raised when the LLM judge fails to produce a valid label after retries.

    Carries a `category` so callers can distinguish the failure mode:
    "empty_response", "malformed_response", or "api_error". Never invented a
    label; this is purely a diagnostic.
    """

    def __init__(self, category, message, raw_response=None):
        super().__init__(f"[{category}] {message}")
        self.category = category
        self.raw_response = raw_response


def _truncate(text, limit=300):
    """Truncate raw model output for safe diagnostics (no secrets, bounded)."""
    if text is None:
        return None
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[+{len(text) - limit} chars]"


def judge_relevance(question, chunk_text):
    """Ask the LLM judge whether `chunk_text` is relevant to `question`.

    Returns {"relevant": str, "reason": str}.

    Retries ONCE on any failure (empty response, malformed JSON, or
    API/network/provider error), telling the model what went wrong. If the
    second attempt also fails, raises JudgeError with a failure category and
    diagnostic detail — never an invented label.
    """
    base_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CHUNK:\n{chunk_text}\n\n"
        f"Return only the JSON object."
    )
    client = _build_client()

    last_error = None
    for attempt in (1, 2):
        prompt = base_prompt
        if attempt == 2 and last_error is not None:
            prompt += (
                f"\n\nYour previous response was invalid: {last_error}. "
                f"Return exactly one JSON object containing BOTH a 'relevant' "
                f"field ('relevant' or 'not_relevant') AND a non-empty 'reason' "
                f"field."
            )
        try:
            response = client.invoke(
                [SystemMessage(content=RUBRIC), HumanMessage(content=prompt)]
            )
            raw = response.content
            if isinstance(raw, list):
                raw = "".join(part.get("text", "") for part in raw)

            if not raw or not raw.strip():
                raise JudgeError(
                    "empty_response",
                    "model returned an empty response",
                    raw_response=raw,
                )

            try:
                return _parse_judgment(raw)
            except ValueError as e:
                raise JudgeError(
                    "malformed_response",
                    str(e),
                    raw_response=_truncate(raw),
                ) from e

        except JudgeError as e:
            # A parse/empty failure from this attempt; captured for retry.
            if attempt == 2:
                raise
            last_error = f"{e.category}: {e}"
        except Exception as e:
            # API/network/provider error (e.g. HTTP 403, connection failure).
            if attempt == 2:
                raise JudgeError(
                    "api_error",
                    f"{type(e).__name__}: {e}",
                ) from e
            last_error = f"{type(e).__name__}: {e}"
    raise JudgeError("other", "unreachable")  # pragma: no cover