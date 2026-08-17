"""CLI: index PDFs into Chroma.

Usage (baseline, unchanged):
    python scripts/ingest.py

Usage (experiment with an isolated store):
    python scripts/ingest.py --chunk-size 800 --chunk-overlap 100 \
        --persist-dir data/chroma_db/experiments/large_800_100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_documents
from src.config import PDF_DIR
from src.ingestion import load_pdfs
from src.vectorstore import add_documents


def main():
    parser = argparse.ArgumentParser(description="Index PDFs into Chroma.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in characters (default: 50)",
    )
    parser.add_argument(
        "--persist-dir",
        default=None,
        help=(
            "Chroma persist directory (default: the baseline store "
            "data/chroma_db). Use an experiments/ path to keep the baseline "
            "store untouched."
        ),
    )
    args = parser.parse_args()

    documents = load_pdfs(PDF_DIR)
    print(f"{len(documents)} pages loaded")
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"{len(chunks)} chunks created")
    add_documents(chunks, persist_dir=args.persist_dir)
    target = args.persist_dir or "data/chroma_db (baseline)"
    print(f"indexed into Chroma ({target})")


if __name__ == "__main__":
    main()
