"""Embedding logic: wraps the sentence embedding model."""

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings(model_name=DEFAULT_EMBEDDING_MODEL):
    """Return the HuggingFaceEmbeddings wrapper for `model_name`.

    Defaults to all-MiniLM-L6-v2 (the frozen baseline). Pass a different
    model name for a controlled embedding experiment. The same model must
    be used for indexing and querying a given store.
    """
    return HuggingFaceEmbeddings(model_name=model_name)