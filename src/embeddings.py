"""Embedding logic with process-level, thread-safe model caching.

The embedding model is part of the frozen retrieval path. Construction is
lazy and keyed by model name so alternate experiment models remain isolated.
"""

from threading import RLock

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embedding_cache = {}
_embedding_lock = RLock()


def get_embeddings(model_name=DEFAULT_EMBEDDING_MODEL):
    """Return the HuggingFaceEmbeddings wrapper for `model_name`.

    Defaults to all-MiniLM-L6-v2 (the frozen baseline). Pass a different
    model name for a controlled embedding experiment. The same model must
    be used for indexing and querying a given store. Instances are created
    once per process/model name and reused across requests.
    """
    with _embedding_lock:
        if model_name not in _embedding_cache:
            _embedding_cache[model_name] = HuggingFaceEmbeddings(
                model_name=model_name
            )
        return _embedding_cache[model_name]