"""Evaluation: run fixed-dataset retrieval and persist top-K results (no LLM)."""

import json
from pathlib import Path

from src.config import EVAL_TOP_K
from src.embeddings import DEFAULT_EMBEDDING_MODEL
from src.vectorstore import get_vectorstore


def _get_vectorstore(persist_dir, model_name=DEFAULT_EMBEDDING_MODEL):
    """Chroma store for a given persist directory (mirrors src.vectorstore)."""
    return get_vectorstore(persist_dir, model_name=model_name)


def run_retrieval(dataset, persist_dir, top_k=EVAL_TOP_K, model_name=DEFAULT_EMBEDDING_MODEL):
    """Run retrieval for every dataset question; return one record per chunk.

    Each record follows the schema frozen in Phase 12.3. `score` is the raw
    L2 distance from `similarity_search_with_score` (lower = more similar);
    rank 1 always has the smallest score. `relevant` starts as `null` and is
    set by manual labeling (12.5). `model_name` must match the model used to
    build the store being queried.
    """
    vectorstore = _get_vectorstore(persist_dir, model_name=model_name)
    records = []
    for question in dataset["questions"]:
        hits = vectorstore.similarity_search_with_score(question["question"], k=top_k)
        for rank, (doc, score) in enumerate(hits, start=1):
            metadata = doc.metadata
            records.append(
                {
                    "question_id": question["id"],
                    "question_text": question["question"],
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": metadata.get("chunk_id"),
                    "document_id": doc.id,
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "page_label": metadata.get("page_label"),
                    "section": metadata.get("section"),
                    "chunk_text": doc.page_content,
                    "relevant": None,
                }
            )
    return records


def write_results(records, path):
    """Write result records as JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record) + "\n" for record in records)
