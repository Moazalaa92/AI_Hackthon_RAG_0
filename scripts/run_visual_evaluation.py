"""Run deterministic visual page retrieval over a fixed evaluation dataset.

This opt-in CLI reads only the annotated question set and a saved Pixel RAG
index. It makes no LLM calls and does not alter the frozen text pipeline or
its artifacts.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR
from src.visual_retrieval import (
    DEFAULT_INDEX,
    embed_query,
    load_index,
    retrieve_pages,
    verify_maxsim_agreement,
)


def read_dataset(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pixel RAG page retrieval.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--name", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    dataset = read_dataset(args.dataset)
    index = load_index(args.index)
    question = dataset["questions"][0]["question"]
    processor_scores, reference_scores = verify_maxsim_agreement(
        embed_query(question, index.model),
        index,
    )
    print(
        "MaxSim agreement on first two pages: "
        f"processor={processor_scores.tolist()} "
        f"reference={reference_scores.tolist()}"
    )

    output_path = Path(RESULTS_DIR) / f"{args.name}_visual_pages.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for question in dataset["questions"]:
            ranked = retrieve_pages(question["question"], index, args.top_k)
            for page in ranked:
                record = {
                    "question_id": question["id"],
                    "question_text": question["question"],
                    "rank": page["rank"],
                    "score": page["score"],
                    "page_label": page["page_label"],
                    "model": page["model"],
                    "dpi": page["dpi"],
                }
                handle.write(json.dumps(record) + "\n")
                count += 1
            print(f"Retrieved {len(ranked)} pages for {question['id']}", flush=True)
    print(f"Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()
