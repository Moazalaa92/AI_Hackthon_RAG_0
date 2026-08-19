"""LLM judges for generation evaluation (Phase 8.5).

Evaluates generated answers on four SEPARATE dimensions. Each dimension is a
distinct LLM call with its OWN prompt and its OWN narrow input set, so no
judgment can leak information from another dimension:

  A. correctness  : question + reference answer + generated answer
                    (NO retrieved context - judge the facts, not the evidence)
  B. grounding    : question + retrieved context + generated answer
                    (NO reference answer - judge only against the evidence)
  C. completeness : question + reference answer + retrieved context + answer
  D. abstention   : question + retrieved context + generated answer
                    (only for deliberate negative questions; reference is not
                    given, because it is intentionally absent)

Follows the same client / retry / error pattern as src/judge.py (Phase 12.5):
same model, temperature 0, one JSON object, one retry on failure.
"""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

CORRECTNESS_RUBRIC = """\
You are an expert annotator evaluating whether a generated answer is factually \
correct against a reference answer extracted from a source document.

You will receive:
- QUESTION: the question that was asked
- REFERENCE ANSWER: the expected facts, extracted from the source document
- GENERATED ANSWER: the answer produced by a RAG system

Judge whether the GENERATED ANSWER is factually correct according to the \
REFERENCE ANSWER. Do not penalize phrasing differences - the same fact stated \
in different words is still correct.

Labels:
- "correct": the answer states the correct fact(s) and no incorrect fact.
- "partially_correct": the answer states part of the correct fact(s) but \
misses key detail(s), or mixes correct facts with a minor inaccuracy.
- "incorrect": the answer states a wrong fact, or substantially misses the \
reference answer.

Important rules:
1. Judge ONLY against the REFERENCE ANSWER. Ignore any context, and ignore \
the question's wording beyond what it asks.
2. If the reference answer requires several items (e.g. a list of medicines) \
and the generated answer gives only some of them, that is "partially_correct" \
or "incorrect" depending on how much is missing and whether anything is wrong.
3. Do not judge the style, length or elegance of the answer.

You MUST respond with a single JSON object and nothing else, in exactly this \
format:
{"label": "correct" or "partially_correct" or "incorrect", "reason": "one \
short sentence explaining the decision"}
"""

GROUNDING_RUBRIC = """\
You are an expert annotator evaluating whether a generated answer is grounded \
in - i.e. supported by - the retrieved context provided to it. This evaluates \
FAITHFULNESS TO THE EVIDENCE, not factual correctness against the real world.

You will receive:
- QUESTION: the question that was asked
- RETRIEVED CONTEXT: the text chunks the system was given
- GENERATED ANSWER: the answer produced by the RAG system

Judge whether every claim in the GENERATED ANSWER is supported by the \
RETRIEVED CONTEXT.

Labels:
- "grounded": every claim in the answer is supported by the retrieved context.
- "partially_grounded": most claims are supported, but some claim(s) go \
beyond the context (e.g. adds a detail not present in the context).
- "unsupported": the answer makes claims that are not supported by the \
retrieved context, or invents details/facts not present in the context.

Important rules:
1. A claim is supported only if the retrieved context actually contains it \
or directly implies it. Do not use outside knowledge: if the context does not \
contain a fact but the fact happens to be true, it is still UNSUPPORTED.
2. "The context does not contain enough information to answer" (a refusal) is \
grounded if the context truly lacks the answer.
3. Do not judge whether the answer is correct or complete - only whether it \
stays within the evidence.

You MUST respond with a single JSON object and nothing else, in exactly this \
format:
{"label": "grounded" or "partially_grounded" or "unsupported", "reason": "one \
short sentence explaining the decision"}
"""

COMPLETENESS_RUBRIC = """\
You are an expert annotator evaluating whether a generated answer sufficiently \
answers the user's question, given the available evidence.

You will receive:
- QUESTION: the question that was asked
- REFERENCE ANSWER: the expected facts, extracted from the source document
- RETRIEVED CONTEXT: the text chunks the system was given
- GENERATED ANSWER: the answer produced by the RAG system

Judge whether the GENERATED ANSWER covers what the question asks, to the \
extent the evidence allows.

Labels:
- "complete": the answer addresses the full question using the available \
evidence.
- "partial": the answer addresses part of the question but misses \
significant points that the evidence supports.
- "incomplete": the answer misses most or all of the substance of the \
question, even though the evidence contains it.

Important rules:
1. A short but complete answer is "complete". Length is irrelevant.
2. If the question asks for a list (e.g. of risk factors) and the answer gives \
only one item when the evidence lists several, that is "partial" or \
"incomplete".
3. If the reference answer is null / the question is not answerable, label \
"not_applicable".
4. Do not penalize the answer for lacking facts that are NOT in the retrieved \
context.

You MUST respond with a single JSON object and nothing else, in exactly this \
format:
{"label": "complete" or "partial" or "incomplete" or "not_applicable", \
"reason": "one short sentence explaining the decision"}
"""

ABSTENTION_RUBRIC = """\
You are an expert annotator evaluating whether a RAG system correctly refused \
to answer a question for which the source document does NOT contain the \
answer.

You will receive:
- QUESTION: the question that was asked
- RETRIEVED CONTEXT: the text chunks the system was given
- GENERATED ANSWER: the answer produced by the RAG system

The question is INTENTIONALLY UNANSWERABLE from the provided document: the \
document does not contain the requested information. The correct behavior is \
for the system to refuse - i.e. say the answer was not found in the provided \
documents (or equivalent).

Labels:
- "appropriate_refusal": the system refused, or answered that the information \
was not found in the provided documents, without inventing an answer.
- "inappropriate_answer": the system produced an answer that invents facts \
not in the context, or answers as if it knew the information.

Important rules:
1. A refusal is appropriate even if phrased differently, as long as it does \
not provide a fabricated answer.
2. If the answer repeats only what the context says without answering the \
actual question and without fabricating, treat it as an appropriate refusal.

You MUST respond with a single JSON object and nothing else, in exactly this \
format:
{"label": "appropriate_refusal" or "inappropriate_answer", "reason": "one \
short sentence explaining the decision"}
"""


def _build_client():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        max_tokens=2000,
    )


def _extract_json(raw_text):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in response: {raw_text!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e


def _validate(data, label_field, allowed):
    if not isinstance(data, dict):
        raise ValueError(f"JSON is not an object: {data!r}")
    label = data.get(label_field)
    reason = data.get("reason")
    if label not in allowed:
        raise ValueError(f"'{label_field}' must be one of {allowed}, got {label!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"'reason' must be a non-empty string, got {reason!r}")
    return {"label": label, "reason": reason.strip()}


class JudgeError(Exception):
    """Raised when a generation judge fails after retries. Never invents a label."""

    def __init__(self, category, message):
        super().__init__(f"[{category}] {message}")
        self.category = category


def _judge(rubric, prompt):
    """Run one judge call with retries; return {label, reason}."""
    client = _build_client()
    last_error = None
    for attempt in (1, 2, 3):
        text = prompt
        if attempt > 1 and last_error is not None:
            text += (
                f"\n\nYour previous response was invalid: {last_error}. "
                f"Return exactly one JSON object with a valid label and a "
                f"non-empty 'reason'."
            )
        try:
            response = client.invoke(
                [SystemMessage(content=rubric), HumanMessage(content=text)]
            )
            raw = response.content
            if isinstance(raw, list):
                raw = "".join(part.get("text", "") for part in raw)
            if not raw or not raw.strip():
                raise JudgeError("empty_response", "model returned an empty response")
            data = _extract_json(raw)
            return data
        except (JudgeError, ValueError) as e:
            if attempt == 3:
                raise JudgeError("malformed_response", str(e)) from e
            last_error = str(e)
        except Exception as e:
            if attempt == 3:
                raise JudgeError("api_error", f"{type(e).__name__}: {e}") from e
            last_error = f"{type(e).__name__}: {e}"
    raise JudgeError("other", "unreachable")  # pragma: no cover


def judge_correctness(question, reference, answer):
    """Judge factual correctness against a reference answer.

    Receives ONLY: question, reference answer, generated answer.
    Receives NO retrieved context (avoids evidence leakage into the verdict).
    """
    data = _judge(
        CORRECTNESS_RUBRIC,
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        "Return only the JSON object.",
    )
    return _validate(data, "label", ("correct", "partially_correct", "incorrect"))


def judge_grounding(question, context, answer):
    """Judge faithfulness of the answer to the retrieved context.

    Receives ONLY: question, retrieved context, generated answer.
    Receives NO reference answer (the judge must not use outside knowledge).
    """
    data = _judge(
        GROUNDING_RUBRIC,
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        "Return only the JSON object.",
    )
    return _validate(data, "label", ("grounded", "partially_grounded", "unsupported"))


def judge_completeness(question, reference, context, answer):
    """Judge whether the answer sufficiently covers the question.

    Receives: question, reference answer, retrieved context, generated answer.
    """
    data = _judge(
        COMPLETENESS_RUBRIC,
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        "Return only the JSON object.",
    )
    return _validate(
        data, "label", ("complete", "partial", "incomplete", "not_applicable")
    )


def judge_abstention(question, context, answer):
    """Judge whether a deliberate-negative question was correctly refused.

    Receives ONLY: question, retrieved context, generated answer.
    The question is intentionally unanswerable from the document.
    """
    data = _judge(
        ABSTENTION_RUBRIC,
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED CONTEXT:\n{context}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        "Return only the JSON object.",
    )
    return _validate(data, "label", ("appropriate_refusal", "inappropriate_answer"))