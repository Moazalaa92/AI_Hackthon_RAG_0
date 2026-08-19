"""Phase 10 STEP 5 — LLM-assisted residual ranking failure diagnosis.

Builds a diagnostic dataset of (question, chunk) pairs for every affected
question (benchmark + holdout), then asks the LLM to classify each chunk into a
strict usefulness taxonomy (DIRECT_ANSWER / SUPPORTING_DETAIL /
RELEVANT_NOT_ANSWERING / RATIONALE / INDEX_TOC / WRONG_SUBTOPIC / SCOPE_MISMATCH
/ OTHER).

Scope per question:
  - all chunks in the reranked Top-10 (the visible retrieval window)
  - PLUS every chunk labeled `relevant` in the candidate pool beyond Top-10
    (captures the "evidence present but ranked low" phenomenon)

The LLM judgment is a SEPARATE diagnostic signal. It does NOT replace or
overwrite the existing frozen relevance labels.

Outputs (all new files under evaluation/experiments/):
  step5_diagnosis.jsonl        raw LLM judgments (strict schema)
  step5_diagnosis_inputs.jsonl the exact inputs sent to the judge (reproducible)
  step5_summary.json           aggregate counts + comparisons

Usage:
  .venv/bin/python evaluation/experiments/step5_diagnose.py --run-llm
  # rerun aggregation only from saved inputs without calling the LLM:
  .venv/bin/python evaluation/experiments/step5_diagnose.py
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

BENCH_CANDS = "evaluation/results/reranked_hybrid_800_100_candidates.jsonl"
BENCH_LABELS = "evaluation/results/hybrid_800_100_candidates_labeled.jsonl"
HOLDOUT_CANDS = "evaluation/results/holdout_v1_reranked_hybrid_candidates.jsonl"
HOLDOUT_LABELS = "evaluation/results/holdout_v1_hybrid_candidates_labeled.jsonl"

OUTDIR = "evaluation/experiments"
DIAG_OUT = os.path.join(OUTDIR, "step5_diagnosis.jsonl")
INPUTS_OUT = os.path.join(OUTDIR, "step5_diagnosis_inputs.jsonl")
SUMMARY_OUT = os.path.join(OUTDIR, "step5_summary.json")

RUBRIC = """\
You are a diagnostic annotator for a medical-guideline retrieval system. You are \
given a QUESTION and a single RETRIEVED CHUNK from a NICE epilepsy guideline. \
Your job is to classify how USEFUL this chunk would be for answering the EXACT \
question. This is NOT a clinical-correctness check and NOT a simple \
topic-similarity check.

Categories (pick exactly one):
- DIRECT_ANSWER: the chunk contains the actual answer the question asks for \
  (the exact medicine, criterion, time-frame, definition, etc.).
- SUPPORTING_DETAIL: the chunk does not contain the headline answer itself but \
  provides directly relevant supporting evidence/details that help justify or \
  flesh out the answer.
- RELEVANT_NOT_ANSWERING: the chunk is topically on the same disease/topic but \
  does not answer this exact question (e.g. discusses the syndrome generally, \
  management in a different situation).
- RATIONALE: the chunk contains committee discussion, "why the committee...", \
  evidence-review rationale, "how recommendations might affect practice", or \
  research-priority statements rather than the recommendation itself.
- INDEX_TOC: table of contents, dot-leader lines, section heading lists, \
  evidence-review index lists (pointers to other sections, not content).
- WRONG_SUBTOPIC: the chunk addresses a clearly different aspect/subtopic of the \
  question (e.g. a different seizure type, a different treatment line, a \
  different population) that a reasonable reader would NOT use as evidence for \
  the exact question.
- SCOPE_MISMATCH: the chunk is phrased almost identically to the question but a \
  qualifier ("with other seizure types", "second-line", "if first-line is \
  unsuccessful", "in children", "not due to X", etc.) narrows or changes the \
  meaning so it does not answer the question as asked.
- OTHER: anything that does not fit the above.

Also answer six yes/no questions:
- direct_answer (bool): does the chunk DIRECTLY answer the exact question?
- supporting_detail (bool): does it provide supporting evidence for the answer?
- topically_relevant (bool): is it on-topic / topically relevant at all?
- rationale_or_index (bool): is it rationale/index/research-context rather than \
  a recommendation?
- scope_mismatch (bool): does a scope/qualifier change the meaning vs the \
  question as asked?
- answer_usefulness: "high" | "medium" | "low" — how useful would this chunk be \
  if given to an answer-generation model for THIS exact question?

Rules:
1. "Relevant" must NOT automatically mean "directly useful". A chunk can be \
   topically relevant and still be RATIONALE, INDEX_TOC, WRONG_SUBTOPIC, or \
   SCOPE_MISMATCH.
2. Judge ONLY from the question and chunk text provided. No external knowledge.
3. DIRECT_ANSWER requires the chunk to actually contain the requested answer \
   content (e.g. if the question asks which medicine is first-line, the chunk \
   must state that medicine, not merely mention the disease).

Respond with ONLY a single JSON object, exactly this shape, and nothing else:
{"direct_answer": true|false, "supporting_detail": true|false,
 "topically_relevant": true|false, "rationale_or_index": true|false,
 "scope_mismatch": true|false, "answer_usefulness": "high"|"medium"|"low",
 "diagnostic_class": "DIRECT_ANSWER", "reason": "one short sentence"}
"""


def _build_client():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        max_tokens=700,
    )


def _parse(raw_text):
    text = raw_text.strip()
    if not text:
        raise ValueError("empty response")
    # find the last complete JSON object (handles truncated/trailing text)
    start = text.find("{")
    end = -1
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if start == -1 or end == -1:
        raise ValueError(f"no complete JSON object in response: {text[:120]!r}")
    data = json.loads(text[start : end + 1])
    valid = {
        "DIRECT_ANSWER", "SUPPORTING_DETAIL", "RELEVANT_NOT_ANSWERING",
        "RATIONALE", "INDEX_TOC", "WRONG_SUBTOPIC", "SCOPE_MISMATCH", "OTHER",
    }
    cls = data.get("diagnostic_class")
    if cls not in valid:
        raise ValueError(f"bad diagnostic_class: {cls!r}")
    if data.get("answer_usefulness") not in ("high", "medium", "low"):
        raise ValueError(f"bad answer_usefulness: {data.get('answer_usefulness')!r}")
    for k in ("direct_answer", "supporting_detail", "topically_relevant",
              "rationale_or_index", "scope_mismatch"):
        if not isinstance(data.get(k), bool):
            raise ValueError(f"bad bool field {k}: {data.get(k)!r}")
    if not isinstance(data.get("reason"), str) or not data["reason"].strip():
        raise ValueError("missing reason")
    return data


def judge(question, chunk_text):
    client = _build_client()
    prompt = f"QUESTION:\n{question}\n\nRETRIEVED CHUNK:\n{chunk_text}\n\nReturn only the JSON object."
    last = None
    for attempt in (1, 2):
        p = prompt
        if attempt == 2 and last is not None:
            p += f"\n\nYour previous response was invalid ({last}). Return exactly one valid JSON object."
        try:
            resp = client.invoke(
                [SystemMessage(content=RUBRIC), HumanMessage(content=p)]
            )
            raw = resp.content
            if isinstance(raw, list):
                raw = "".join(part.get("text", "") for part in raw)
            return _parse(raw)
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"judge failed after retries: {last}")


def load_candidates(path, labpath):
    can = {}
    lab = {}
    for line in open(path):
        r = json.loads(line)
        can[(r["question_id"], r["chunk_id"])] = r
    for line in open(labpath):
        r = json.loads(line)
        lab[(r["question_id"], r["chunk_id"])] = r["relevant"]
    return can, lab


def affected_questions(can, lab):
    """Questions where a relevant chunk sits below the top-3 (P@3 loss)."""
    affected = []
    for q in sorted(set(k[0] for k in can)):
        rows = [k for k in can if k[0] == q]
        rows.sort(key=lambda k: can[k]["reranker_rank"])
        top3 = rows[:3]
        rel3 = sum(1 for k in top3 if lab[k] == "relevant")
        missed = [k for k in rows[3:] if lab[k] == "relevant"]
        if rel3 < 3 and missed:
            affected.append(q)
    return affected


def build_inputs(can, lab, questions):
    """Top-10 + all relevant-labeled chunks beyond top-10, per question."""
    inputs = []
    for q in questions:
        rows = [k for k in can if k[0] == q]
        rows.sort(key=lambda k: can[k]["reranker_rank"])
        top10 = rows[:10]
        beyond_relevant = [k for k in rows[10:] if lab[k] == "relevant"]
        seen = set()
        for k in top10 + beyond_relevant:
            if k in seen:
                continue
            seen.add(k)
            r = can[k]
            inputs.append({
                "question_id": q,
                "question_text": r["question_text"],
                "chunk_id": r["chunk_id"],
                "chunk_text": r["chunk_text"],
                "reranker_rank": r["reranker_rank"],
                "reranker_score": r["reranker_score"],
                "dense_rank": r["dense_rank"],
                "bm25_rank": r["bm25_rank"],
                "fusion_rank": r["rank"],
                "page_label": r["page_label"],
                "existing_relevance": lab[k],
                "in_top10": k in set(top10),
            })
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-llm", action="store_true", help="call the LLM judge")
    ap.add_argument("--limit", type=int, default=0, help="max inputs to judge (debug)")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    bench_can, bench_lab = load_candidates(BENCH_CANDS, BENCH_LABELS)
    hold_can, hold_lab = load_candidates(HOLDOUT_CANDS, HOLDOUT_LABELS)

    bench_q = affected_questions(bench_can, bench_lab)
    hold_q = affected_questions(hold_can, hold_lab)
    print("affected benchmark:", sorted(bench_q))
    print("affected holdout:  ", sorted(hold_q))

    inputs = build_inputs(bench_can, bench_lab, bench_q) + \
             build_inputs(hold_can, hold_lab, hold_q)

    # always save inputs for reproducibility
    with open(INPUTS_OUT, "w") as f:
        for item in inputs:
            f.write(json.dumps(item) + "\n")
    print(f"saved {len(inputs)} diagnostic inputs -> {INPUTS_OUT}")

    if args.run_llm:
        if args.limit:
            inputs = inputs[: args.limit]
        # resume: skip inputs already successfully judged in a previous run;
        # re-judge any that ended in JUDGE_ERROR (e.g. earlier truncation bug)
        done = set()
        if os.path.exists(DIAG_OUT):
            for line in open(DIAG_OUT):
                try:
                    r = json.loads(line)
                    if "JUDGE_ERROR" in r["judgment"]["reason"]:
                        continue
                    done.add((r["question_id"], r["chunk_id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        remaining = [i for i in inputs if (i["question_id"], i["chunk_id"]) not in done]
        print(f"already judged: {len(done)}  remaining: {len(remaining)}")
        with open(DIAG_OUT, "a") as f:
            for i, item in enumerate(remaining, 1):
                try:
                    j = judge(item["question_text"], item["chunk_text"])
                except Exception as e:  # noqa: BLE001
                    j = {"diagnostic_class": "OTHER", "answer_usefulness": "low",
                         "direct_answer": False, "supporting_detail": False,
                         "topically_relevant": False, "rationale_or_index": False,
                         "scope_mismatch": False, "reason": f"JUDGE_ERROR: {e}"}
                rec = dict(item)
                rec["judgment"] = j
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if i % 25 == 0:
                    print(f"  judged {i}/{len(remaining)}")
        print(f"saved LLM judgments -> {DIAG_OUT}")


if __name__ == "__main__":
    main()