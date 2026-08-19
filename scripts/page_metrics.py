"""Compute deterministic page-level retrieval metrics without LLM calls.

This script evaluates either visual page rankings or text chunk rankings. Text
rankings are collapsed to first-occurrence page order, while ground truth is
either the dataset's annotated pages or pages marked relevant in an existing
labeled candidate pool. Frozen text artifacts and labels are read-only.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR

K_VALUES = (1, 3, 5, 10)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def page_label(record):
    label = record.get("page_label")
    if label is None and record.get("page") is not None:
        label = str(record["page"] + 1)
    if label is None:
        raise ValueError(f"Record has no page identifier: {record!r}")
    return str(label)


def ranked_pages(records):
    """Return page labels in rank order for each question.

    Duplicate chunk pages remain in the sequence so a Top-K chunk ranking is
    collapsed inside each requested K window, matching the frozen baseline.
    """
    by_question = defaultdict(list)
    for record in records:
        order = record.get("reranker_rank")
        if order is None:
            order = record.get("rank")
        if order is None:
            raise ValueError(f"Record has no rank: {record!r}")
        by_question[record["question_id"]].append((order, record))

    output = {}
    for question_id, pairs in by_question.items():
        pairs.sort(key=lambda item: item[0])
        output[question_id] = [page_label(record) for _, record in pairs]
    return output


def ground_truth(dataset, mode, pool_path=None):
    if mode == "annotated":
        return {
            question["id"]: {str(page) for page in question.get("expected_pages", [])}
            for question in dataset["questions"]
        }
    if not pool_path:
        raise ValueError("--pool is required for judged ground truth")
    labels = defaultdict(set)
    for record in read_jsonl(pool_path):
        if record.get("relevant") == "relevant":
            labels[record["question_id"]].add(page_label(record))
    return dict(labels)


def compute_metrics(ranking, truth, question_ids):
    usable = [question_id for question_id in question_ids if truth.get(question_id)]
    excluded = len(question_ids) - len(usable)
    result = {
        "n_questions": len(usable),
        "excluded_empty_ground_truth": excluded,
        "metrics": {},
    }
    if not usable:
        result["metrics"] = {
            f"PageHit@{k}": None for k in K_VALUES
        } | {
            f"PageCoverage@{k}": None for k in K_VALUES
        }
        return result
    for k in K_VALUES:
        hits = []
        coverage = []
        for question_id in usable:
            retrieved = set(ranking.get(question_id, [])[:k])
            expected = truth[question_id]
            hits.append(bool(retrieved & expected))
            coverage.append(len(retrieved & expected) / len(expected))
        result["metrics"][f"PageHit@{k}"] = sum(hits) / len(usable)
        result["metrics"][f"PageCoverage@{k}"] = sum(coverage) / len(usable)
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute offline page metrics.")
    parser.add_argument("--input", required=True, help="Visual page or text chunk JSONL")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--ground-truth", choices=("annotated", "judged"), required=True
    )
    parser.add_argument("--pool", help="Labeled candidate pool for judged mode")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    dataset = read_json(args.dataset)
    question_ids = [question["id"] for question in dataset["questions"]]
    ranking = ranked_pages(read_jsonl(args.input))
    truth = ground_truth(dataset, args.ground_truth, args.pool)
    payload = {
        "name": args.name,
        "input": args.input,
        "dataset": args.dataset,
        "ground_truth": args.ground_truth,
        **compute_metrics(ranking, truth, question_ids),
    }
    output_path = Path(METRICS_DIR) / f"{args.name}_pages.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
