"""CLI: generate answers for the Phase 8.5 generation evaluation.

Reads the FROZEN canonical retrieval results read-only and runs the existing
Generation implementation (src/generation.py) for every question in
evaluation/generation_dataset_v1.json. No retrieval is performed here, no
retrieval artifact is modified.

Canonical retrieval inputs (the validated dense+BM25 -> cross-encoder Top-10):
    evaluation/results/reranked_hybrid_800_100_top10_labeled.jsonl  (Q01-Q20)
    evaluation/results/holdout_v1_reranked_hybrid_top10_labeled.jsonl (H01-H27)

Output:
    evaluation/results/generation_v1.jsonl   (one JSON object per question)

Resumable: questions already present in the output file are skipped on re-run.

Usage:
    python scripts/run_generation_evaluation.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR
from src.generation import generate_answer

BENCHMARK_RETRIEVAL = "reranked_hybrid_800_100_top10_labeled.jsonl"
HOLDOUT_RETRIEVAL = "holdout_v1_reranked_hybrid_top10_labeled.jsonl"
DATASET_FILE = "evaluation/generation_dataset_v1.json"
OUTPUT_FILE = "generation_v1.jsonl"


def read_jsonl(path):
    """Load all JSON objects from a JSONL file; exit on missing/malformed input."""
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


def load_retrieval_by_question():
    """Group frozen Top-10 records by question_id, ranked by reranker_rank.

    Returns {qid: [record, ...]} where records are sorted by reranker_rank
    (the final ranking signal of the frozen architecture).
    """
    by_question = {}
    for path in (BENCHMARK_RETRIEVAL, HOLDOUT_RETRIEVAL):
        full = Path(RESULTS_DIR) / path
        if not full.exists():
            sys.exit(f"Error: frozen retrieval results not found: {full}")
        for record in read_jsonl(full):
            by_question.setdefault(record["question_id"], []).append(record)
    for qid in by_question:
        by_question[qid].sort(key=lambda r: r["reranker_rank"])
    return by_question


def load_done_questions(path):
    """Question IDs already present in the output file (for resumability)."""
    done = set()
    if not Path(path).exists():
        return done
    for record in read_jsonl(path):
        if record.get("question_id"):
            done.add(record["question_id"])
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Generate answers over the frozen retrieval Top-10 (Phase 8.5)."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=600,
        help="Maximum tokens for each generated answer (default: 600)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(RESULTS_DIR) / OUTPUT_FILE),
        help=f"Output JSONL path (default: {Path(RESULTS_DIR) / OUTPUT_FILE})",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    retrieval = load_retrieval_by_question()
    done = load_done_questions(out_path)

    questions = dataset["questions"]
    skipped = 0
    generated = 0
    failed = 0
    generated_at = datetime.now(timezone.utc).isoformat()

    with open(out_path, "a", encoding="utf-8") as out:
        for q in questions:
            qid = q["id"]
            if qid in done:
                skipped += 1
                continue

            chunks = retrieval.get(qid)
            if chunks is None:
                print(
                    f"  ! no frozen retrieval results for {qid} (skipping)",
                    file=sys.stderr,
                )
                failed += 1
                continue

            try:
                answer = generate_answer(
                    q["question"], chunks, max_tokens=args.max_tokens
                )
            except Exception as e:
                failed += 1
                print(f"  ! generation failed for {qid}: {e}", file=sys.stderr)
                continue

            record = {
                "question_id": qid,
                "question": q["question"],
                "answerable": q["answerable"],
                "source_dataset": q.get("source_dataset"),
                "generated_answer": answer,
                "generated_at": generated_at,
                "max_tokens": args.max_tokens,
                "retrieved_chunk_ids": [r["chunk_id"] for r in chunks],
                "retrieved_chunk_pages": [
                    {"chunk_id": r["chunk_id"], "page": r.get("page_label"),
                     "reranker_rank": r["reranker_rank"]}
                    for r in chunks
                ],
                "retrieved_chunks": chunks,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            done.add(qid)
            generated += 1
            print(f"  {qid}: generated ({len(answer)} chars)")

    print(f"\nQuestions in dataset:       {len(questions)}")
    print(f"Already generated (skipped): {skipped}")
    print(f"Generated this run:          {generated}")
    print(f"Failed:                      {failed}")
    print(f"Output written to:           {out_path}")


if __name__ == "__main__":
    main()