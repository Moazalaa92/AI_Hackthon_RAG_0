"""CLI: label a results file by REUSING labels from a labeled candidate pool.

A reranking experiment only re-orders an existing candidate pool, so every
(question, chunk) pair it returns has already been judged. Re-running the LLM
judge on those pairs costs money and injects judge variance (~5% of labels flip
between runs), which is indistinguishable from a real ranking change. This
script copies the existing label instead, keyed on
(question_id, document_id).

Usage:
    python scripts/apply_pool_labels.py \
        evaluation/results/reranked_ensemble_800_100_top10.jsonl \
        --pool evaluation/results/hybrid_800_100_candidates_labeled.jsonl

Writes <name>_labeled.jsonl next to the input file. Exits with an error if any
record has no label in the pool (that means the experiment changed the pool and
genuinely needs the judge).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_records(path):
    """Load all JSON objects from a JSONL file; exit on a missing/bad file."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Copy relevance labels from a labeled candidate pool."
    )
    parser.add_argument("results_file", help="Unlabeled results JSONL to label")
    parser.add_argument(
        "--pool",
        required=True,
        help="Labeled candidate-pool JSONL to copy labels from",
    )
    args = parser.parse_args()

    labels = {}
    for record in read_records(args.pool):
        if record.get("relevant") is not None:
            labels[(record["question_id"], record["document_id"])] = record["relevant"]

    records = read_records(args.results_file)
    missing = [
        (r["question_id"], r["document_id"])
        for r in records
        if (r["question_id"], r["document_id"]) not in labels
    ]
    if missing:
        sys.exit(
            f"Error: {len(missing)} record(s) have no label in the pool, e.g. "
            f"{missing[:3]}. These pairs were never judged — run "
            f"scripts/label_results.py instead."
        )

    out_path = Path(args.results_file)
    out_path = out_path.with_name(f"{out_path.stem}_labeled.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            labeled = dict(record)
            labeled["relevant"] = labels[(record["question_id"], record["document_id"])]
            labeled["label_source"] = Path(args.pool).name
            f.write(json.dumps(labeled) + "\n")

    print(f"Reused {len(records)} labels from {args.pool}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
