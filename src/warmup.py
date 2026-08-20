"""Explicit startup warm-up for the frozen retrieval and reranking path."""

from src.embeddings import DEFAULT_EMBEDDING_MODEL
from src.reranking import CROSS_ENCODER_MODEL, get_reranker
from src.vectorstore import get_vectorstore


def warm_up(
    persist_dir,
    *,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    reranker_model=CROSS_ENCODER_MODEL,
):
    """Load the configured embedding, vectorstore, and reranker handles.

    The returned handles are the same process-level cached instances used by
    normal requests. This is intended for an API startup hook so first-user
    latency does not pay model construction costs.
    """
    vectorstore = get_vectorstore(persist_dir, model_name=embedding_model)
    reranker = get_reranker(reranker_model)
    return vectorstore, reranker
