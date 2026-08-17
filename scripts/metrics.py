"""CLI: Precision@3 and Precision@5 from a labeled results file (Phase 12.6/12.7).

Reads ONLY the labeled JSONL produced by Phase 12.5. No LLM, no retrieval,
no Chroma. Never modifies the input file.

Usage:
    python scripts/metrics.py evaluation/results/baseline_500_50_top10_labeled.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR
from src.metrics import precision_at_k, relevant_at_10, validate_labeled


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


def experiment_name(results_path):
    """Derive the experiment name from the filename, e.g.
    baseline_500_50_top10_labeled.jsonl -> baseline_500_50."""
    return Path(results_path).stem.replace("_top10_labeled", "").replace("_labeled", "")


def write_metrics_json(out_path, experiment, k, per_question, mean, caveat):
    payload = {
        "experiment": experiment,
        "k": k,
        "mean_precision": round(mean, 6),
        "per_question": {
            qid: {"precision": round(v["precision"], 6), "relevant": v["relevant"], "n": v["n"]}
            for qid, v in per_question.items()
        },
        "caveat": caveat,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute Precision@3 and Precision@5 from labeled results."
    )
    parser.add_argument(
        "labeled_file",
        help="Path to the labeled JSONL results file (Phase 12.5 output)",
    )
    args = parser.parse_args()

    records = read_records(args.labeled_file)
    if not records:
        sys.exit(f"Error: no records found in {args.labeled_file}")

    try:
        validate_labeled(records)
    except ValueError as e:
        sys.exit(f"Error: invalid labeled set: {e}")

    p3, mean3 = precision_at_k(records, k=3)
    p5, mean5 = precision_at_k(records, k=5)
    rel10 = relevant_at_10(records)

    caveat = (
        "LLM-judged retrieval relevance (deepseek/deepseek-v4-flash, temp 0). "
        "These are NOT clinical ground truth; they are a proxy for retrieval quality."
    )

    print("Per-question results (LLM-judged retrieval relevance):")
    print(f"{'Q':>4} {'Rel@3':>6} {'P@3':>7} {'Rel@5':>6} {'P@5':>7} {'Rel@10':>7}")
    for qid in sorted(p3):
        r3 = p3[qid]["relevant"]
        r5 = p5[qid]["relevant"]
        print(
            f"{qid:>4} {r3:>6} {p3[qid]['precision']:>7.4f} "
            f"{r5:>6} {p5[qid]['precision']:>7.4f} {rel10[qid]:>7}"
        )
    print(f"\nMean Precision@3 = {mean3:.4f}")
    print(f"Mean Precision@5 = {mean5:.4f}")
    print(f"\n{'-' * 60}")
    print(f"Note: {caveat}")

    exp = experiment_name(args.labeled_file)
    out3 = write_metrics_json(
        Path(METRICS_DIR) / f"{exp}_p3.json", exp, 3, p3, mean3, caveat
    )
    out5 = write_metrics_json(
        Path(METRICS_DIR) / f"{exp}_p5.json", exp, 5, p5, mean5, caveat
    )
    print(f"\nWrote {out3}")
    print(f"Wrote {out5}")


if __name__ == "__main__":
    main()
