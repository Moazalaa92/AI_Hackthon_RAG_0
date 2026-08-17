"""CLI: Top-3 vs Top-5 vs Top-10 analysis (Phase 12.8).

Rank-distribution analysis of relevant chunks from the labeled results file.
Reads ONLY the labeled JSONL produced by Phase 12.5. No LLM, no retrieval,
no Chroma. Never modifies the input file.

Usage:
    python scripts/analyze_topk.py evaluation/results/baseline_500_50_top10_labeled.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR
from src.metrics import topk_analysis


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


def write_topk_json(out_path, experiment, per_question, aggregates, caveat):
    payload = {
        "experiment": experiment,
        "per_question": per_question,
        "aggregates": aggregates,
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
        description="Top-3 vs Top-5 vs Top-10 rank analysis from labeled results."
    )
    parser.add_argument(
        "labeled_file",
        help="Path to the labeled JSONL results file (Phase 12.5 output)",
    )
    args = parser.parse_args()

    records = read_records(args.labeled_file)
    if not records:
        sys.exit(f"Error: no records found in {args.labeled_file}")

    per_question, aggregates = topk_analysis(records)

    caveat = (
        "LLM-judged retrieval relevance (deepseek/deepseek-v4-flash, temp 0). "
        "These are NOT clinical ground truth; they are a proxy for retrieval quality."
    )

    print("Top-K analysis (per question):")
    print(
        f"{'Q':>4} {'1st_rel':>7} {'Rel@3':>6} {'Rel@5':>6} {'Rel@10':>7} "
        f"{'P@3':>7} {'P@5':>7} {'P@10':>7} {'ranks4-5':>18} {'ranks6-10':>18}"
    )
    for qid in per_question:
        a = per_question[qid]
        first = a["first_relevant_rank"] if a["first_relevant_rank"] is not None else "-"
        r45 = "/".join("R" if x == "relevant" else "X" for x in a["ranks_4_5_relevance"])
        r610 = "/".join("R" if x == "relevant" else "X" for x in a["ranks_6_10_relevance"])
        print(
            f"{qid:>4} {first:>7} {a['relevant_count_at_3']:>6} "
            f"{a['relevant_count_at_5']:>6} {a['relevant_count_at_10']:>7} "
            f"{a['precision_at_3']:>7.4f} {a['precision_at_5']:>7.4f} "
            f"{a['precision_at_10']:>7.4f} {r45:>18} {r610:>18}"
        )

    agg = aggregates
    dist = agg["first_relevant_rank_distribution"]
    print("\nAggregate statistics (across 20 questions):")
    print(f"  questions with >=1 relevant in Top-3: {agg['questions_with_relevant_at_3']} "
          f"({agg['pct_with_relevant_at_3']:.1%})")
    print(f"  questions with >=1 relevant in Top-5: {agg['questions_with_relevant_at_5']} "
          f"({agg['pct_with_relevant_at_5']:.1%})")
    print(f"  questions with >=1 relevant in Top-10: {agg['questions_with_relevant_at_10']} "
          f"({agg['pct_with_relevant_at_10']:.1%})")
    print(f"  questions with NO relevant in Top-10: {agg['questions_with_no_relevant_at_10']}")
    print("  first relevant rank distribution:")
    for rank in range(1, 11):
        print(f"    rank {rank:>2} -> {dist[str(rank)]} questions")
    print(f"    none  -> {dist['none']} questions")

    print(f"\n{'-' * 60}")
    print(f"Note: {caveat}")

    exp = experiment_name(args.labeled_file)
    out = write_topk_json(
        Path(METRICS_DIR) / f"{exp}_topk.json", exp, per_question, aggregates, caveat
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
