"""Retrieval logic: question -> ranked top-K chunks (no LLM involved)."""

from src.config import TOP_K
from src.vectorstore import get_vectorstore


def retrieve(query, top_k=TOP_K):
    """Return (Document, score) pairs ranked by similarity distance."""
    return get_vectorstore().similarity_search_with_score(query, k=top_k)