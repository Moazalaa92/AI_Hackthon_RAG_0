"""CLI: index PDFs into Chroma."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_documents
from src.config import PDF_DIR
from src.ingestion import load_pdfs
from src.vectorstore import add_documents


def main():
    documents = load_pdfs(PDF_DIR)
    print(f"{len(documents)} pages loaded")
    chunks = chunk_documents(documents)
    print(f"{len(chunks)} chunks created")
    add_documents(chunks)
    print("indexed into Chroma")


if __name__ == "__main__":
    main()