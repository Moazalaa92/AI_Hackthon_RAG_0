"""Vector store logic with process-level, thread-safe handle caching."""

from threading import RLock

import chromadb
from langchain_chroma import Chroma

from src.config import CHROMA_DB_DIR
from src.embeddings import DEFAULT_EMBEDDING_MODEL, get_embeddings

_vectorstore_cache = {}
_chroma_client_cache = {}
_vectorstore_lock = RLock()


def get_vectorstore(persist_dir=None, model_name=DEFAULT_EMBEDDING_MODEL):
    """Return a Chroma vector store instance (persisted).

    `persist_dir` defaults to the baseline store (CHROMA_DB_DIR); pass an
    experiment directory to query an isolated index. Collection name stays
    "documents" in every directory — isolation comes from the directory.
    `model_name` must match the model used to build the store being queried.
    Handles are created once per `(persist_dir, model_name)` and reused across
    requests.
    """
    store_path = str(persist_dir or CHROMA_DB_DIR)
    key = (store_path, model_name)
    with _vectorstore_lock:
        if store_path not in _chroma_client_cache:
            _chroma_client_cache[store_path] = chromadb.PersistentClient(
                path=store_path
            )
        if key not in _vectorstore_cache:
            _vectorstore_cache[key] = Chroma(
                collection_name="documents",
                persist_directory=store_path,
                embedding_function=get_embeddings(model_name),
                client=_chroma_client_cache[store_path],
            )
        return _vectorstore_cache[key]


def add_documents(documents, persist_dir=None, model_name=DEFAULT_EMBEDDING_MODEL):
    """Index chunks into the vector store.

    Uses chunk_id as the document id so re-indexing overwrites instead of
    duplicating. `persist_dir` defaults to the baseline store; pass an
    experiment directory to keep experiments isolated. `model_name` is the
    embedding model used to vectorize the chunks.
    """
    vectorstore = get_vectorstore(persist_dir, model_name=model_name)
    ids = [f"chunk_{doc.metadata['chunk_id']}" for doc in documents]
    vectorstore.add_documents(documents, ids=ids)
    return vectorstore
