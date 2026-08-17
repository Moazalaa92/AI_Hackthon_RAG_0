"""End-to-end pipeline: PDF -> chunks -> vector store -> retrieve top-K."""

from src.chunking import chunk_documents
from src.config import PDF_DIR, TOP_K
from src.ingestion import load_pdfs
from src.retrieval import retrieve
from src.vectorstore import add_documents, get_vectorstore


def main():
    documents = load_pdfs(PDF_DIR)
    print(f"{len(documents)} pages loaded")

    chunks = chunk_documents(documents)
    print(f"{len(chunks)} chunks created")

    add_documents(chunks)
    print(f"indexed (total in store: {len(get_vectorstore().get()['ids'])})")

    question = input("Ask a question (Enter for default): ").strip()
    if not question:
        question = "Dravet syndrome treatment"

    print(f"\nTop-{TOP_K} for: {question!r}")
    for i, (doc, score) in enumerate(retrieve(question, top_k=TOP_K), start=1):
        meta = doc.metadata
        print(f'  #{i}  {score:.4f}  chunk_{meta.get("chunk_id")}  p.{meta.get("page")}')
        print(f'      "{doc.page_content[:200]}..."')


if __name__ == "__main__":
    main()