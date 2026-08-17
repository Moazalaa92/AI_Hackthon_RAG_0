"""Chunking logic for splitting documents into retrievable pieces."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks, preserving metadata and adding a chunk_id."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
