"""Phase 6: inspect the vector store - insert chunks, search, confirm persistence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_documents
from src.config import PDF_DIR
from src.ingestion import load_pdfs
from src.vectorstore import add_documents, get_vectorstore


def main():
    documents = load_pdfs(PDF_DIR)
    chunks = chunk_documents(documents)
    print(f"{len(chunks)} chunks created")

    store = get_vectorstore()
    existing = len(store.get()["ids"])
    print(f"chunks already in store: {existing}")

    add_documents(chunks)
    print(f"collection count after insert: {len(store.get()['ids'])}")

    print("\nSearching...")
    for query in [
        "Dravet syndrome treatment",
        "what medicines are recommended",
        "MRI scan",
    ]:
        print(f"\nQuery: {query!r}")
        for doc, score in store.similarity_search_with_score(query, k=5):
            meta = doc.metadata
            print(
                f"  score={score:.4f} "
                f"chunk_id={meta.get('chunk_id')} "
                f"page={meta.get('page')}"
            )
            print(f"    {doc.page_content[:120]}...")


if __name__ == "__main__":
    main()