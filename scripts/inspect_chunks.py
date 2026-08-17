"""Inspection logic for examining documents"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def inspect_chunks(chunks, limit=5):
    """Inspect chunks: id, length, first/last 100 chars, page."""
    print(f"\nInspecting first {limit} chunks:")
    for i, chunk in enumerate(chunks[:limit]):
        print(f"Chunk {i + 1}:")
        print(f"  chunk_id: {chunk.metadata.get('chunk_id')}")
        print(f"  Page: {chunk.metadata.get('page')}")
        print(f"  Length: {len(chunk.page_content)}")
        print(f"  Start: {chunk.page_content[:100]}...")
        print(f"  End: ...{chunk.page_content[-100:]}")
        print()


if __name__ == "__main__":
    from src.chunking import chunk_documents
    from src.config import PDF_DIR
    from src.ingestion import load_pdfs

    documents = load_pdfs(PDF_DIR)

    chunks = chunk_documents(documents)
    print(f"\n{len(chunks)} chunks created")
    inspect_chunks(chunks)
