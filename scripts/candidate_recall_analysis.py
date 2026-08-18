"""CLI: candidate-recall analysis for the hybrid retrieval experiment.

Reads ONLY the labeled candidate-pool file produced by labeling
`<name>_candidates.jsonl` (Phase: hybrid experiment). No LLM, no retrieval,
no Chroma. Never modifies the input file.

For every question it reports whether a relevant chunk exists in the Dense
Top-20, the BM25 Top-20, and the union, plus the per-retriever provenance of
every relevant chunk (dense-only / bm25-only / both). This is the primary
evidence for whether BM25 expands the candidate pool that dense retrieval
misses.

Usage:
    python scripts/candidate_recall_analysis.py \
        evaluation/results/hybrid_800_100_candidates_labeled.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR


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
            records.append(record)
    return records


def analyze(records):
    """Per-question candidate recall using the labeled candidate pool."""
    by_question = {}
    for record in records:
        by_question.setdefault(record["question_id"], []).append(record)

    per_question = {}
    for qid, question_records in sorted(by_question.items()):
        relevant = [
            r for r in question_records if r["relevant"] == "relevant"
        ]
        dense_top20 = any(r["dense_rank"] is not None for r in relevant)
        bm25_top20 = any(r["bm25_rank"] is not None for r in relevant)
        per_question[qid] = {
            "question_id": qid,
            "union_size": len(question_records),
            "relevant_in_dense_top20": dense_top20,
            "relevant_in_bm25_top20": bm25_top20,
            "relevant_in_union": len(relevant) > 0,
            "relevant_chunks": [
                {
                    "document_id": r["document_id"],
                    "page_label": r["page_label"],
                    "dense_rank": r["dense_rank"],
                    "bm25_rank": r["bm25_rank"],
                    "fusion_rank": r["rank"],
                    "retrieved_by": r["retrieved_by"],
                }
                for r in relevant
            ],
        }

    aggregates = {
        "questions_with_relevant_in_dense_top20": sum(
            1 for q in per_question.values() if q["relevant_in_dense_top20"]
        ),
        "questions_with_relevant_in_bm25_top20": sum(
            1 for q in per_question.values() if q["relevant_in_bm25_top20"]
        ),
        "questions_with_relevant_in_union": sum(
            1 for q in per_question.values() if q["relevant_in_union"]
        ),
        "questions_with_no_relevant_in_either_top20": sum(
            1 for q in per_question.values() if not q["relevant_in_union"]
        ),
        "relevant_chunk_provenance": {
            "dense_only": sum(
                1
                for q in per_question.values()
                for c in q["relevant_chunks"]
                if c["dense_rank"] is not None and c["bm25_rank"] is None
            ),
            "bm25_only": sum(
                1
                for q in per_question.values()
                for c in q["relevant_chunks"]
                if c["dense_rank"] is None and c["bm25_rank"] is not None
            ),
            "both": sum(
                1
                for q in per_question.values()
                for c in q["relevant_chunks"]
                if c["dense_rank"] is not None and c["bm25_rank"] is not None
            ),
        },
    }
    return per_question, aggregates


def main():
    parser = argparse.ArgumentParser(
        description="Candidate-recall analysis for the hybrid retrieval experiment."
    )
    parser.add_argument(
        "labeled_candidates_file",
        help="Path to the labeled candidate-pool JSONL file",
    )
    args = parser.parse_args()

    records = read_records(args.labeled_candidates_file)
    per_question, aggregates = analyze(records)

    print("Candidate recall per question (Dense Top-20 vs BM25 Top-20 vs union):")
    print(
        f"{'Q':>4} {'dense20':>8} {'bm2520':>8} {'union':>6} "
        f"{'union_size':>11} {'rel_chunks':>10}"
    )
    for q in per_question.values():
        print(
            f"{q['question_id']:>4} {str(q['relevant_in_dense_top20']):>8} "
            f"{str(q['relevant_in_bm25_top20']):>8} {str(q['relevant_in_union']):>6} "
            f"{q['union_size']:>11} {len(q['relevant_chunks']):>10}"
        )

    a = aggregates
    print("\nAggregates (across 20 questions):")
    print(f"  relevant found in Dense Top-20 : {a['questions_with_relevant_in_dense_top20']}")
    print(f"  relevant found in BM25 Top-20  : {a['questions_with_relevant_in_bm25_top20']}")
    print(f"  relevant found in union        : {a['questions_with_relevant_in_union']}")
    print(f"  no relevant in either Top-20   : {a['questions_with_no_relevant_in_either_top20']}")
    p = a["relevant_chunk_provenance"]
    print(
        f"  relevant chunks: dense-only={p['dense_only']} "
        f"bm25-only={p['bm25_only']} both={p['both']}"
    )

    exp = Path(args.labeled_candidates_file).stem.replace("_candidates_labeled", "")
    out_path = Path(METRICS_DIR) / f"{exp}_candidate_recall.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": exp,
        "per_question": per_question,
        "aggregates": aggregates,
        "caveat": (
            "LLM-judged retrieval relevance (deepseek/deepseek-v4-flash, temp 0). "
            "Candidate pool = union of Dense Top-20 and BM25 Top-20 before fusion."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()