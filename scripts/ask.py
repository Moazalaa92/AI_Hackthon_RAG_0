"""CLI: ask a question end-to-end (retrieval -> generation) for Phase 8.

Pipeline:
    question
      ↓
    canonical retrieval (dense Top-20 + BM25 Top-20 -> cross-encoder -> Top-10)
      ↓
    generation (grounded answer from the retrieved chunks)
      ↓
    print answer

Reuses the existing retrieval modules (src.hybrid_retrieval + src.reranking);
no retrieval logic is duplicated here.

Usage:
    python scripts/ask.py "Which medicine is offered as first-line treatment for absence seizures?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings import DEFAULT_EMBEDDING_MODEL
from src.generation import generate_answer
from src.hybrid_retrieval import BM25_CANDIDATES, DENSE_CANDIDATES, run_hybrid
from src.reranking import rerank_candidates

DEFAULT_STORE = "data/chroma_db/experiments/large_800_100"
DEFAULT_FINAL_K = 10


def retrieve_top_k(question, persist_dir=DEFAULT_STORE, final_k=DEFAULT_FINAL_K):
    """Run the canonical frozen retrieval for a single question.

    Returns the final Top-K chunk records (dicts) ranked by the cross-encoder,
    reusing src.hybrid_retrieval.run_hybrid and src.reranking.rerank_candidates.
    """
    dataset = {"questions": [{"id": "ASK", "question": question}]}
    _, candidates = run_hybrid(
        dataset,
        persist_dir=persist_dir,
        model_name=DEFAULT_EMBEDDING_MODEL,
        dense_k=DENSE_CANDIDATES,
        bm25_k=BM25_CANDIDATES,
        final_k=final_k,
    )
    reranked = rerank_candidates(candidates)
    return [r for r in reranked if r["reranker_rank"] <= final_k]


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question: canonical retrieval -> grounded answer (Phase 8)."
    )
    parser.add_argument("question", help="The question to answer")
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE,
        help=f"Chroma persist directory to retrieve from (default: {DEFAULT_STORE})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_FINAL_K,
        help=f"Number of chunks to retrieve for context (default: {DEFAULT_FINAL_K})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=600,
        help="Maximum tokens for the generated answer (default: 600)",
    )
    args = parser.parse_args()

    print(f"Retrieving top-{args.top_k} for: {args.question!r}")
    chunks = retrieve_top_k(args.question, persist_dir=args.store, final_k=args.top_k)

    print(f"\nRetrieved {len(chunks)} chunks:")
    for r in chunks:
        meta = r.get("page_label") or r.get("page")
        print(
            f"  rank {r['reranker_rank']:>2}  chunk_{r.get('chunk_id')}  "
            f"p.{meta}  {r.get('source')}  ({r.get('retrieved_by')})"
        )
        print(f'    "{r["chunk_text"][:90]}..."')

    print("\nGenerating answer...\n")
    answer = generate_answer(args.question, chunks, max_tokens=args.max_tokens)
    print(answer)


if __name__ == "__main__":
    main()