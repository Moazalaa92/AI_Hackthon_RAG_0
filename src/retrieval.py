"""Retrieval logic: question -> ranked top-K chunks (no LLM involved)."""

from src.config import TOP_K
from src.vectorstore import get_vectorstore


def retrieve(query, top_k=TOP_K, persist_dir=None):
    """Return (Document, score) pairs ranked by similarity distance.

    `persist_dir` defaults to the baseline store; pass an experiment directory
    to query an isolated index (keeps experiments fair and isolated).
    """
    return get_vectorstore(persist_dir).similarity_search_with_score(query, k=top_k)
