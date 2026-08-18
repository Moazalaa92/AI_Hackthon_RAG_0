"""Retrieval logic: question -> ranked top-K chunks (no LLM involved)."""

from src.config import TOP_K
from src.embeddings import DEFAULT_EMBEDDING_MODEL
from src.vectorstore import get_vectorstore


def retrieve(query, top_k=TOP_K, persist_dir=None, model_name=DEFAULT_EMBEDDING_MODEL):
    """Return (Document, score) pairs ranked by similarity distance.

    `persist_dir` defaults to the baseline store; pass an experiment directory
    to query an isolated index (keeps experiments fair and isolated).
    `model_name` must match the model used to build the store being queried.
    """
    return get_vectorstore(persist_dir, model_name=model_name).similarity_search_with_score(
        query, k=top_k
    )
