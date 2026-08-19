"""Sweep visual page scores as a deterministic prior over text chunks.

The frozen cross-encoder scores each existing labeled candidate pool. A
visual page score is then added after per-question z-standardization. No
candidate generation, labels, or LLM calls are changed; metrics and ceilings
are reused from the existing ranker sweep implementation.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweep_rankers import ceiling, group, metrics, read_records
from src.config import METRICS_DIR
from src.reranking import CROSS_ENCODER_MODEL, score_candidates, zscore
from src.visual_retrieval import (
    DEFAULT_INDEX,
    embed_query,
    load_index,
    score_pages,
)

WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)


def ordered_records(by_question):
    return [
        record
        for question_id in sorted(by_question)
        for record in by_question[question_id]
    ]


def page_label(record):
    if record.get("page_label") is not None:
        return str(record["page_label"])
    return str(record["page"] + 1)


def visual_scores(dataset, index):
    labels = index.page_labels
    output = {}
    for question in dataset["questions"]:
        query = embed_query(question["question"], index.model)
        scores = score_pages(query, index)
        output[question["id"]] = dict(zip(labels, scores))
    return output


def run_pool(pool_path, index, dataset):
    records = read_records(pool_path)
    by_question = group(records)
    ordered = ordered_records(by_question)
    ce_by_question = score_candidates(ordered, CROSS_ENCODER_MODEL)
    visual_by_question = visual_scores(dataset, index)

    def visual_order(question_id, recs):
        return [visual_by_question[question_id][page_label(r)] for r in recs]

    results = {"visual only": metrics(
        {
            question_id: sorted(
                recs,
                key=lambda r: -visual_by_question[question_id][page_label(r)],
            )
            for question_id, recs in by_question.items()
        }
    )}
    for weight in WEIGHTS:
        ranked = {}
        for question_id, recs in by_question.items():
            ce = zscore(ce_by_question[question_id])
            visual = zscore(visual_order(question_id, recs))
            scores = [c + weight * v for c, v in zip(ce, visual)]
            ranked[question_id] = [
                record
                for _, record in sorted(
                    zip(scores, recs), key=lambda item: -item[0]
                )
            ]
        results[f"text CE + {weight}*visual"] = metrics(ranked)

    return {
        "n_questions": len(by_question),
        "ceiling": ceiling(by_question),
        "strategies": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep visual priors over labeled pools.")
    parser.add_argument("pools", nargs="+")
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--holdout-dataset",
        help="Dataset for the second pool; defaults to --dataset for all pools",
    )
    parser.add_argument(
        "--out", default=str(Path(METRICS_DIR) / "visual_prior_sweep.json")
    )
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as handle:
        first_dataset = json.load(handle)
    second_dataset = first_dataset
    if args.holdout_dataset:
        with open(args.holdout_dataset, encoding="utf-8") as handle:
            second_dataset = json.load(handle)
    datasets = [first_dataset, second_dataset]
    index = load_index(args.index)
    summary = {}
    for pool_path, dataset in zip(args.pools, datasets):
        print(f"Sweeping {pool_path}", flush=True)
        summary[Path(pool_path).stem] = run_pool(pool_path, index, dataset)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
