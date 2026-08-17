"""CLI: label frozen retrieval results with the LLM relevance judge (Phase 12.5).

Reads ONLY the frozen JSONL produced by Phase 12.3. Never performs retrieval
(no Chroma, no embeddings, no PDF loading, no chunking). Writes a DERIVATIVE
labeled JSONL; the original results file is never modified.

Usage:
    python scripts/label_results.py evaluation/results/baseline_500_50_top10.jsonl

Behavior:
- Creates <baseline>_labeled.jsonl next to the input file.
- Resumable/idempotent: records already labeled (relevant != null) in the
  labeled file are skipped; a re-run after a partial run finishes the rest.
- Labels are written to the file one record at a time, so an interrupted run
  keeps every completed label.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import LLM_MODEL
from src.judge import JudgeError, judge_relevance


def read_records(path):
    """Load all JSON objects from a JSONL file; exit with a clear error on
    a missing file or a malformed line."""
    if not Path(path).exists():
        sys.exit(f"Error: results file not found: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"Error: invalid JSON on line {lineno} of {path}: {e}")
            if not isinstance(record, dict):
                sys.exit(f"Error: line {lineno} of {path} is not a JSON object")
            records.append(record)
    return records


def labeled_path(results_path):
    """Derivative filename: <name>_labeled.jsonl next to the input file."""
    p = Path(results_path)
    return p.with_name(f"{p.stem}_labeled.jsonl")


def load_labeled_keys(path):
    """Return {(question_id, rank)} already present (and labeled) in a labeled
    file, so a re-run skips them."""
    keys = set()
    if not Path(path).exists():
        return keys
    for record in read_records(path):
        if record.get("relevant") is not None:
            keys.add((record.get("question_id"), record.get("rank")))
    return keys


def main():
    parser = argparse.ArgumentParser(
        description="Label frozen retrieval results with the LLM relevance judge."
    )
    parser.add_argument(
        "results_file",
        help="Path to the JSONL results file produced by Phase 12.3",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    out_path = labeled_path(results_path)

    records = read_records(results_path)
    if not records:
        sys.exit(f"Error: no records found in {results_path}")

    done_keys = load_labeled_keys(out_path)

    skipped = 0
    newly_labeled = 0
    failed = 0
    judged_at = datetime.now(timezone.utc).isoformat()

    # Append-only so an interrupted run keeps every completed label.
    with open(out_path, "a", encoding="utf-8") as out:
        for record in records:
            key = (record.get("question_id"), record.get("rank"))
            if key in done_keys:
                skipped += 1
                continue

            try:
                judgment = judge_relevance(
                    record["question_text"], record["chunk_text"]
                )
            except JudgeError as e:
                failed += 1
                print(
                    f"  ! failed to label {record.get('question_id')} "
                    f"rank {record.get('rank')} [{e.category}]: {e}",
                    file=sys.stderr,
                )
                continue

            labeled = dict(record)
            labeled["relevant"] = judgment["relevant"]
            labeled["judge"] = "llm"
            labeled["judge_model"] = LLM_MODEL
            labeled["judge_reason"] = judgment["reason"]
            labeled["judged_at"] = judged_at

            out.write(json.dumps(labeled) + "\n")
            out.flush()
            done_keys.add(key)
            newly_labeled += 1

    print(f"Records in frozen file:        {len(records)}")
    print(f"Already labeled (skipped):     {skipped}")
    print(f"Newly labeled this run:        {newly_labeled}")
    print(f"Failed (left unlabeled):       {failed}")
    print(f"Judge model:                   {LLM_MODEL}")
    print(f"Labeled results written to:    {out_path}")
    if newly_labeled == 0 and failed == 0:
        print("Nothing to do - all records were already labeled (idempotent).")
    if failed:
        print(
            f"WARNING: {failed} record(s) were left unlabeled. Re-run this "
            f"command to retry them (the process is resumable).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()