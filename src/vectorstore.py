"""Vector store logic for persisting and querying embedded documents."""
from langchain_chroma import Chroma

from src.config import CHROMA_DB_DIR
from src.embeddings import get_embeddings


def get_vectorstore():
    """Return a Chroma vector store instance (persisted)."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name="documents",
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )


def add_documents(documents):
    """Index chunks into the vector store.

    Uses chunk_id as the document id so re-indexing overwrites instead of
    duplicating.
    """
    vectorstore = get_vectorstore()
    ids = [f"chunk_{doc.metadata['chunk_id']}" for doc in documents]
    vectorstore.add_documents(documents, ids=ids)
    return vectorstore