"""Evaluation: run fixed-dataset retrieval and persist top-K results (no LLM)."""

import json
from pathlib import Path

from langchain_chroma import Chroma

from src.config import EVAL_TOP_K
from src.embeddings import get_embeddings


def _get_vectorstore(persist_dir):
    """Chroma store for a given persist directory (mirrors src.vectorstore)."""
    return Chroma(
        collection_name="documents",
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(),
    )


def run_retrieval(dataset, persist_dir, top_k=EVAL_TOP_K):
    """Run retrieval for every dataset question; return one record per chunk.

    Each record follows the schema frozen in Phase 12.3. `score` is the raw
    L2 distance from `similarity_search_with_score` (lower = more similar);
    rank 1 always has the smallest score. `relevant` starts as `null` and is
    set by manual labeling (12.5).
    """
    vectorstore = _get_vectorstore(persist_dir)
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
        for record in records:
            f.write(json.dumps(record) + "\n")
