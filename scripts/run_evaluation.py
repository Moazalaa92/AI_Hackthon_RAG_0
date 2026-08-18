"""CLI: run retrieval for the evaluation dataset against a store (no LLM).

Usage:
    python scripts/run_evaluation.py --store data/chroma_db
    python scripts/run_evaluation.py --store data/chroma_db --name baseline_500_50
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EVAL_TOP_K, EVALUATION_DIR, RESULTS_DIR
from src.embeddings import DEFAULT_EMBEDDING_MODEL
from src.evaluation import run_retrieval, write_results


def main():
    parser = argparse.ArgumentParser(description="Run retrieval on the eval dataset.")
    parser.add_argument(
        "--store",
        default="data/chroma_db",
        help="Chroma persist directory to query (default: data/chroma_db)",
    )
    parser.add_argument(
        "--name",
        default="baseline_500_50",
        help="Experiment name used in the output filename (default: baseline_500_50)",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(EVALUATION_DIR) / "dataset.json"),
        help="Path to the evaluation dataset (default: evaluation/dataset.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=EVAL_TOP_K,
        help=f"Number of chunks to retrieve per question (default: {EVAL_TOP_K})",
    )
    parser.add_argument(
        "--embedding",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            f"Sentence-transformer model name (default: {DEFAULT_EMBEDDING_MODEL}). "
            "Must match the model used to build the store being queried."
        ),
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    records = run_retrieval(
        dataset, persist_dir=args.store, top_k=args.top_k, model_name=args.embedding
    )

    out_path = Path(RESULTS_DIR) / f"{args.name}_top{args.top_k}.jsonl"
    write_results(records, out_path)

    questions = len(dataset["questions"])
    print(f"Ran {questions} questions x top-{args.top_k} against store: {args.store}")
    print(f"Embedding model: {args.embedding}")
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
