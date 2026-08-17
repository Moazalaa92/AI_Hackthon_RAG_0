"""Ingestion logic for processing documents"""
import os

from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(pdf_directory):
    """Load PDFs from the specified directory and return a list of documents."""
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    return documents
