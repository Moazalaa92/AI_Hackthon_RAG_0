"""CLI: evaluate generated answers with the generation judges (Phase 8.5).

Reads evaluation/results/generation_v1.jsonl (generated answers over the frozen
Top-10), judges each answer on correctness / grounding / completeness /
abstention, and attributes failures to retrieval vs generation.

Leakage control - what each judge receives:
    correctness  : question + reference answer + generated answer   (NO context)
    grounding    : question + retrieved context + generated answer   (NO reference)
    completeness : question + reference + retrieved context + answer
    abstention   : question + retrieved context + generated answer   (negatives only)

Failure attribution (for answerable questions with a wrong/partial answer):
    retrieval_failure : NO relevant chunk in the frozen Top-10 context
    generation_failure: relevant chunk(s) present in Top-10, but the answer
                        is still wrong/partial
Relevance comes from the frozen retrieval labels (relevant != null in the
frozen labeled files); nothing here re-labels retrieval.

Output:
    evaluation/results/generation_v1_judged.jsonl

Resumable: already-judged question_ids are skipped on re-run.

Usage:
    python scripts/evaluate_generation.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR
from src.generation import build_context
from src.generation_judge import (
    JudgeError,
    judge_abstention,
    judge_completeness,
    judge_correctness,
    judge_grounding,
)

GENERATED_FILE = "generation_v1.jsonl"
BENCHMARK_RETRIEVAL = "reranked_hybrid_800_100_top10_labeled.jsonl"
HOLDOUT_RETRIEVAL = "holdout_v1_reranked_hybrid_top10_labeled.jsonl"
DATASET_FILE = "evaluation/generation_dataset_v1.json"
OUTPUT_FILE = "generation_v1_judged.jsonl"


def read_jsonl(path):
    if not Path(path).exists():
        sys.exit(f"Error: file not found: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"Error: invalid JSON on line {lineno} of {path}: {e}")
    return records


def load_dataset():
    with open(DATASET_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_relevant_by_question():
    """Map {qid: set(chunk_ids that were judged relevant)} from frozen labels.

    Frozen labels come from the Phase 12.5/13 LLM relevance judge. A chunk is
    'present evidence' for a question if the frozen judge marked it relevant.
    """
    out = {}
    for path in (BENCHMARK_RETRIEVAL, HOLDOUT_RETRIEVAL):
        full = Path(RESULTS_DIR) / path
        if not full.exists():
            sys.exit(f"Error: frozen retrieval results not found: {full}")
        for record in read_jsonl(full):
            if record.get("relevant") == "relevant":
                out.setdefault(record["question_id"], set()).add(record["chunk_id"])
    return out


def load_done_ids(path):
    done = set()
    if not Path(path).exists():
        return done
    for record in read_jsonl(path):
        if record.get("question_id"):
            done.add(record["question_id"])
    return done


def attribute_failure(qid, answerable, retrieved_chunk_ids, relevant_by_q,
                      correctness_label):
    """Classify a wrong/partial answer as retrieval or generation failure.

    - No relevant evidence in Top-10 -> retrieval_failure.
    - Relevant evidence present but answer still wrong/partial -> generation_failure.
    - Correct answer -> None.
    """
    if not answerable or correctness_label == "correct":
        return None
    relevant = relevant_by_q.get(qid, set())
    present = any(cid in relevant for cid in retrieved_chunk_ids)
    return "generation_failure" if present else "retrieval_failure"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers with the generation judges."
    )
    parser.add_argument(
        "--input",
        default=str(Path(RESULTS_DIR) / GENERATED_FILE),
        help=f"Generated answers JSONL (default: {Path(RESULTS_DIR) / GENERATED_FILE})",
    )
    parser.add_argument(
        "--output",
        default=str(Path(RESULTS_DIR) / OUTPUT_FILE),
        help=f"Judged output JSONL (default: {Path(RESULTS_DIR) / OUTPUT_FILE})",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    dataset_by_id = {q["id"]: q for q in dataset["questions"]}
    relevant_by_q = load_relevant_by_question()
    done = load_done_ids(out_path)

    records = read_jsonl(in_path)
    if not records:
        sys.exit(f"Error: no records in {in_path}")

    skipped = 0
    judged = 0
    failed = 0
    judged_at = datetime.now(timezone.utc).isoformat()

    with open(out_path, "a", encoding="utf-8") as out:
        for record in records:
            qid = record["question_id"]
            if qid in done:
                skipped += 1
                continue

            answerable = record["answerable"]
            reference = dataset_by_id.get(qid, {}).get("reference")
            answer = record["generated_answer"]
            context = build_context(record["retrieved_chunks"])

            entry = {
                "question_id": qid,
                "question": record["question"],
                "answerable": answerable,
                "generated_answer": answer,
                "retrieved_chunk_ids": record["retrieved_chunk_ids"],
            }

            try:
                if answerable:
                    correctness = judge_correctness(
                        record["question"], reference, answer
                    )
                    grounding = judge_grounding(
                        record["question"], context, answer
                    )
                    completeness = judge_completeness(
                        record["question"], reference, context, answer
                    )
                    entry["correctness"] = correctness["label"]
                    entry["grounding"] = grounding["label"]
                    entry["completeness"] = completeness["label"]
                    entry["abstention"] = "not_applicable"
                    entry["judge_reasons"] = {
                        "correctness": correctness["reason"],
                        "grounding": grounding["reason"],
                        "completeness": completeness["reason"],
                    }
                else:
                    entry["correctness"] = "not_applicable"
                    entry["grounding"] = "not_applicable"
                    entry["completeness"] = "not_applicable"
                    abstention = judge_abstention(
                        record["question"], context, answer
                    )
                    entry["abstention"] = abstention["label"]
                    entry["judge_reasons"] = {"abstention": abstention["reason"]}
            except JudgeError as e:
                failed += 1
                print(f"  ! judge failed for {qid} [{e.category}]: {e}",
                      file=sys.stderr)
                continue

            entry["failure_type"] = attribute_failure(
                qid, answerable, record["retrieved_chunk_ids"], relevant_by_q,
                entry["correctness"],
            )
            entry["evidence_present"] = any(
                cid in relevant_by_q.get(qid, set())
                for cid in record["retrieved_chunk_ids"]
            ) if answerable else None
            entry["judge"] = "llm"
            entry["judged_at"] = judged_at

            out.write(json.dumps(entry) + "\n")
            out.flush()
            done.add(qid)
            judged += 1

    print(f"Records in generated file:     {len(records)}")
    print(f"Already judged (skipped):      {skipped}")
    print(f"Judged this run:               {judged}")
    print(f"Failed (left unjudged):        {failed}")
    print(f"Judged results written to:     {out_path}")
    if failed:
        print(
            f"WARNING: {failed} record(s) unjudged. Re-run to retry (resumable).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()