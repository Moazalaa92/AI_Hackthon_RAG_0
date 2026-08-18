"""CLI: rerank the hybrid candidate pool with a cross-encoder (no RRF).

Reads ONLY the established hybrid candidate pool
(evaluation/results/hybrid_800_100_candidates.jsonl). No retrieval, no
chunking, no embedding model, no BM25 rebuild — the candidate-generation stage
is frozen. The cross-encoder scores every (question, chunk) pair in the union
and the union is sorted by that score; the final Top-K is the cross-encoder's
ranking. `rank`/`score` are set to the cross-encoder ranking; raw provenance
(dense_rank/bm25_rank/dense_score/bm25_score/fusion_score) is preserved.

Usage:
    python scripts/run_rerank_evaluation.py \
        --candidates evaluation/results/hybrid_800_100_candidates.jsonl \
        --name reranked_hybrid_800_100

Outputs:
    evaluation/results/<name>_top10.jsonl         final Top-10 (200 records)
    evaluation/results/<name>_candidates.jsonl    full union + reranker_score/rank
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR
from src.evaluation import write_results
from src.reranking import CROSS_ENCODER_MODEL, rerank_candidates


def read_records(path):
    """Load all JSON objects from a JSONL file; exit on a missing/malformed file."""
    if not Path(path).exists():
        sys.exit(f"Error: candidates file not found: {path}")
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
        description="Rerank the hybrid candidate pool with a cross-encoder."
    )
    parser.add_argument(
        "--candidates",
        default=str(Path(RESULTS_DIR) / "hybrid_800_100_candidates.jsonl"),
        help="Candidate-pool JSONL from the hybrid experiment (default: hybrid_800_100)",
    )
    parser.add_argument(
        "--name",
        default="reranked_hybrid_800_100",
        help="Experiment name used in output filenames (default: reranked_hybrid_800_100)",
    )
    parser.add_argument(
        "--model",
        default=CROSS_ENCODER_MODEL,
        help=f"Cross-encoder model name (default: {CROSS_ENCODER_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Final Top-K after reranking (default: 10)",
    )
    args = parser.parse_args()

    records = read_records(args.candidates)
    reranked = rerank_candidates(records, model_name=args.model)

    for r in reranked:
        r["score"] = r["reranker_score"]
        r["rank"] = r["reranker_rank"]

    topk = [r for r in reranked if r["reranker_rank"] <= args.top_k]

    topk_path = Path(RESULTS_DIR) / f"{args.name}_top{args.top_k}.jsonl"
    full_path = Path(RESULTS_DIR) / f"{args.name}_candidates.jsonl"
    write_results(topk, topk_path)
    write_results(reranked, full_path)

    print(f"Reranked {len(reranked)} candidate records with {args.model}")
    print(f"Wrote {len(topk)} final records to {topk_path}")
    print(f"Wrote {len(reranked)} reranked candidate records to {full_path}")


if __name__ == "__main__":
    main()