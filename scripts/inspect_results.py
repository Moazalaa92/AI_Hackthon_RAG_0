"""CLI: display the frozen retrieval results for one question (read-only).

Reads ONLY the persisted JSONL results produced by Phase 12.3. It performs
no retrieval: no Chroma access, no embeddings, no LLM, no rerunning of the
evaluation. It never modifies the results file.

Data flow:
    Phase 12.3 JSONL
            |
            v
    read records
            |
            v
    filter by question_id
            |
            v
    sort by rank
            |
            v
    format
            |
            v
    terminal output

Usage:
    python scripts/inspect_results.py evaluation/results/baseline_500_50_top10.jsonl Q01
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

DISPLAY_FIELDS = ("question_id", "rank", "score", "chunk_text")


def read_records(path):
    """Load every JSON object from a JSONL file. Exit with a clear error on
    a missing file or a malformed line."""
    if not Path(path).exists():
        sys.exit(f"Error: results file not found: {path}")

    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"Error: invalid JSON on line {lineno} of {path}: {e}")
            if not isinstance(record, dict):
                sys.exit(f"Error: line {lineno} of {path} is not a JSON object")
            records.append(record)
    return records


def records_for_question(records, question_id):
    """Keep only the records whose question_id matches (in file order)."""
    return [r for r in records if r.get("question_id") == question_id]


def missing_display_fields(record):
    """Return the list of required display fields absent from a record."""
    return [f for f in DISPLAY_FIELDS if f not in record]


def format_score(score):
    """Format the raw L2 distance. Lower = closer in embedding space; it is
    NOT a percentage and must not be converted into one."""
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def chunk_display_id(record):
    """Human-facing chunk id: prefer document_id (e.g. 'chunk_602')."""
    document_id = record.get("document_id")
    if document_id:
        return document_id
    chunk_id = record.get("chunk_id")
    if chunk_id is not None:
        return f"chunk_{chunk_id}"
    return "?"


def format_record(record):
    """Render one record as a header line plus the full, indented chunk text."""
    rank = record.get("rank")
    score = format_score(record.get("score"))
    source = record.get("source") or "?"

    page_label = record.get("page_label")
    page_display = f"p.{page_label}" if page_label is not None else "p.?"

    section = record.get("section")
    section_display = section if section is not None else "N/A"

    header = (
        f"  #{rank}  {score}  {chunk_display_id(record)}  {source}  "
        f"{page_display}  section: {section_display}"
    )
    body = textwrap.indent(record.get("chunk_text", ""), "      ")
    return header + "\n" + body


def main():
    parser = argparse.ArgumentParser(
        description="Display the frozen retrieval results for one question (read-only)."
    )
    parser.add_argument(
        "results_file",
        help="Path to the JSONL results file produced by Phase 12.3",
    )
    parser.add_argument(
        "question_id",
        help="Question ID to inspect, e.g. Q01",
    )
    args = parser.parse_args()

    records = read_records(args.results_file)
    if not records:
        sys.exit(f"Error: no records found in {args.results_file}")

    matches = records_for_question(records, args.question_id)
    if not matches:
        available = sorted(
            {r.get("question_id") for r in records if r.get("question_id")}
        )
        listed = ", ".join(available) if available else "none"
        sys.exit(
            f"Error: question ID '{args.question_id}' not found in {args.results_file}. "
            f"Available question IDs: {listed}"
        )

    for record in matches:
        missing = missing_display_fields(record)
        if missing:
            rank = record.get("rank")
            sys.exit(
                f"Error: record for question '{args.question_id}' at rank {rank} is "
                f"malformed (missing fields: {', '.join(missing)}). "
                f"Inspect {args.results_file} directly."
            )

    matches.sort(key=lambda r: r.get("rank"))

    question_text = matches[0].get("question_text") or "?"
    print(f"{args.question_id}: {question_text}")
    print()
    for record in matches:
        print(format_record(record))
        print()


if __name__ == "__main__":
    main()
