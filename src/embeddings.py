"""Embedding logic: wraps the sentence embedding model."""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """Return the HuggingFaceEmbeddings wrapper for all-MiniLM-L6-v2."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )