"""CLI: retrieval-only inspection (no LLM involved)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TOP_K
from src.retrieval import retrieve


def main():
    questions = [
        "Dravet syndrome treatment",
        "what medicines are recommended for epilepsy",
        "when is MRI recommended",
        "topiramate in women and girls of childbearing potential",
        "sodium valproate as first-line treatment",
    ]
    for question in questions:
        print(f'\nQ: "{question}"')
        for i, (doc, score) in enumerate(retrieve(question, top_k=TOP_K), start=1):
            meta = doc.metadata
            print(
                f'  #{i}  {score:.4f}  chunk_{meta.get("chunk_id")}  '
                f'{meta.get("source")}  p.{meta.get("page")}'
            )
            print(f'      "{doc.page_content[:80]}..."')


if __name__ == "__main__":
    main()
    