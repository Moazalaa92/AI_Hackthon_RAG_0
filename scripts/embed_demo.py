"""Phase 5 demo: embed a few sentences and print pairwise cosine similarity."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings import get_embeddings


def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(a @ b)


if __name__ == "__main__":
    sentences = [
        "How do I install pip?",
        "What is the procedure to set up pip?",
        "The capital of France is Paris.",
        "Pip is a package manager for Python.",
    ]

    model = get_embeddings()
    vectors = model.embed_documents(sentences)

    print(f"Embedding dimension: {len(vectors[0])}")
    for i, s in enumerate(sentences):
        print(f"  {i}: {s}")
    print()

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"  sim({i}, {j}) = {sim:.3f}")