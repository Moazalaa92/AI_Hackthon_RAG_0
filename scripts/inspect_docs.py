"""Inspection logic for examining documents"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def inspect_documents(documents, limit=5):
    """Inspect the loaded documents, print details, and return a summary."""
    summary = []
    print(f"\nInspecting first {limit} documents:")
    for i, doc in enumerate(documents[:limit]):
        print(f"Document {i + 1}:")
        print(f"  Page: {doc.metadata.get('page')}")
        print(f"  Source: {doc.metadata.get('source')}")
        print(f"  Page Content: {doc.page_content}...")
        print()
        summary.append(
            {
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source"),
                "content": doc.page_content,
            }
        )
    return summary


if __name__ == "__main__":
    from src.config import PDF_DIR
    from src.ingestion import load_pdfs

    documents = load_pdfs(PDF_DIR)
    print(f"{len(documents)} pages loaded")
    summary = inspect_documents(documents)
    print(f"returned {len(summary)} document summaries")
