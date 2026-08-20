"""Evaluate deterministic and optional LLM safety intent classification.

This runner performs intent classification only: it never invokes generation,
retrieval, or an LLM when the layer-2 flag is disabled. It is intentionally
separate from the historical three-way safety evaluation, whose stored NORMAL
cases include frozen generation outputs.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.safety import NORMAL, PATIENT_SPECIFIC, classify_intent

DEFAULT_DATASETS = (
    "evaluation/safety_dataset_v1.json",
    "evaluation/safety_dataset_v2_extra.json",
)


def _load_cases(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if "cases" in data:
        return data["cases"]
    return [
        {
            "id": question["id"],
            "category": NORMAL,
            "expected_outcome": "answer",
            "question": question["question"],
        }
        for question in data["questions"]
    ]


def _evaluate(path):
    records = []
    for case in _load_cases(path):
        result = classify_intent(case["question"])
        records.append(
            {
                "dataset": str(path),
                "id": case["id"],
                "category": case["category"],
                "expected_outcome": case["expected_outcome"],
                "question": case["question"],
                "classification": result.classification,
                "safe_to_answer": result.safe_to_answer,
                "reason": result.reason,
            }
        )
    return records


def _print_report(records):
    print("\nPer-case results")
    print("| Dataset | ID | Actual | Returned | Safe | Question | Reason |")
    print("|---|---|---|---|---|---|---|")
    for record in records:
        print(
            f"| {Path(record['dataset']).name} | {record['id']} | "
            f"{record['category']} | {record['classification']} | "
            f"{'yes' if record['safe_to_answer'] else 'no'} | "
            f"{record['question']} | {record['reason']} |"
        )

    confusion = Counter(
        (record["category"], record["classification"]) for record in records
    )
    print("\nConfusion matrix (actual rows, returned columns)")
    print("| Actual | NORMAL | PATIENT_SPECIFIC |")
    print("|---|---:|---:|")
    actual_categories = [NORMAL, PATIENT_SPECIFIC]
    if any(r["category"] == "INSUFFICIENT_EVIDENCE" for r in records):
        actual_categories.append("INSUFFICIENT_EVIDENCE")
    for actual in actual_categories:
        print(
            f"| {actual} | {confusion[(actual, NORMAL)]} | "
            f"{confusion[(actual, PATIENT_SPECIFIC)]} |"
        )

    patient_cases = [r for r in records if r["category"] == PATIENT_SPECIFIC]
    normal_cases = [r for r in records if r["category"] == NORMAL]
    false_negatives = sum(
        r["classification"] == NORMAL for r in patient_cases
    )
    false_positives = sum(
        r["classification"] == PATIENT_SPECIFIC for r in normal_cases
    )
    print(
        f"\nPATIENT_SPECIFIC -> NORMAL: {false_negatives}/{len(patient_cases)}"
    )
    print(f"NORMAL -> PATIENT_SPECIFIC: {false_positives}/{len(normal_cases)}")
    return false_negatives, false_positives


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the safety intent classifier without generation."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset JSON path; may be supplied multiple times.",
    )
    args = parser.parse_args()
    datasets = args.datasets or list(DEFAULT_DATASETS)
    records = []
    for dataset in datasets:
        records.extend(_evaluate(dataset))
    _print_report(records)


if __name__ == "__main__":
    main()
