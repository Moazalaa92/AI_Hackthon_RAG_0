"""CLI: run hybrid retrieval (dense + BM25 + RRF) for the evaluation dataset.

Dense arm reuses the existing 800/100 + MiniLM store and the same
`similarity_search_with_score` call as the dense-only control. BM25 is built
from the SAME stored chunks. No reranking stage.

Usage:
    python scripts/run_hybrid_evaluation.py \
        --store data/chroma_db/experiments/large_800_100 \
        --name hybrid_800_100

Outputs (JSONL, one record per chunk per question):
    evaluation/results/<name>_top10.jsonl            final Top-10 after fusion
    evaluation/results/<name>_candidates.jsonl       full candidate union (rank 1..n)

Every record carries the frozen schema fields plus retrieval provenance:
retrieved_by, dense_rank, bm25_rank, dense_score, bm25_score, fusion_score.
`score` is the RRF fusion score (higher = better); `dense_score` keeps the raw
L2 distance. `relevant` starts as null (set by label_results.py).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EVALUATION_DIR, RESULTS_DIR
from src.embeddings import DEFAULT_EMBEDDING_MODEL
from src.evaluation import write_results
from src.hybrid_retrieval import (
    BM25_CANDIDATES,
    DENSE_CANDIDATES,
    RRF_K,
    run_hybrid,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run hybrid retrieval (dense + BM25 + RRF) on the eval dataset."
    )
    parser.add_argument(
        "--store",
        default="data/chroma_db",
        help="Chroma persist directory to query (default: data/chroma_db)",
    )
    parser.add_argument(
        "--name",
        default="hybrid_800_100",
        help="Experiment name used in output filenames (default: hybrid_800_100)",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(EVALUATION_DIR) / "dataset.json"),
        help="Path to the evaluation dataset (default: evaluation/dataset.json)",
    )
    parser.add_argument(
        "--embedding",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            f"Sentence-transformer model name (default: {DEFAULT_EMBEDDING_MODEL}). "
            "Must match the model used to build the store being queried."
        ),
    )
    parser.add_argument(
        "--dense-k",
        type=int,
        default=DENSE_CANDIDATES,
        help=f"Dense candidate depth before fusion (default: {DENSE_CANDIDATES})",
    )
    parser.add_argument(
        "--bm25-k",
        type=int,
        default=BM25_CANDIDATES,
        help=f"BM25 candidate depth before fusion (default: {BM25_CANDIDATES})",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=10,
        help="Final Top-K after fusion (default: 10)",
    )
    parser.add_argument(
        "--rrf-k",
        type=float,
        default=RRF_K,
        help=f"RRF smoothing constant k (default: {RRF_K})",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    topk_records, candidate_records = run_hybrid(
        dataset,
        persist_dir=args.store,
        model_name=args.embedding,
        dense_k=args.dense_k,
        bm25_k=args.bm25_k,
        final_k=args.final_k,
        rrf_k=args.rrf_k,
    )

    topk_path = Path(RESULTS_DIR) / f"{args.name}_top{args.final_k}.jsonl"
    candidates_path = Path(RESULTS_DIR) / f"{args.name}_candidates.jsonl"
    write_results(topk_records, topk_path)
    write_results(candidate_records, candidates_path)

    questions = len(dataset["questions"])
    print(f"Ran hybrid retrieval on {questions} questions against store: {args.store}")
    print(f"Embedding model: {args.embedding}")
    print(f"Dense candidates: {args.dense_k} | BM25 candidates: {args.bm25_k} "
          f"| RRF k: {args.rrf_k}")
    print(f"Wrote {len(topk_records)} final records to {topk_path}")
    print(f"Wrote {len(candidate_records)} candidate records to {candidates_path}")


if __name__ == "__main__":
    main()