"""CLI: build claim-level citations over existing generated answers (Phase 9).

Reads the Phase 8.5 generation outputs read-only (evaluation/results/
generation_v2.jsonl, which already embeds the frozen Top-10 retrieved chunks
per question) and builds, for every answer, a list of claims with their
supporting citations using src/sources.py + src/sources_judge.py.

Deterministic guarantees (enforced by src.sources.validate_citations):
    - every citation points to a chunk actually retrieved for that question
    - supporting_text is the exact chunk text (never invented)
    - page / page_label / source are copied from the chunk's own metadata

No retrieval is performed here, no generation artifact is modified, and the
Generation prompt/model are untouched.

Output:
    evaluation/results/citations_v1.jsonl   (one JSON object per question)

Resumable: questions already present in the output file are skipped on re-run.

Usage:
    python scripts/run_citations.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR
from src.sources import build_citations, validate_citations
from src.sources_judge import SourceJudgeError, select_supporting_chunks

GENERATED_FILE = "generation_v2.jsonl"
OUTPUT_FILE = "citations_v1.jsonl"
CITATION_VERSION = "citations_v1"


def read_jsonl(path):
    if not Path(path).exists():
        sys.exit(f"Error: file not found: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"Error: invalid JSON on line {lineno} of {path}: {e}")
    return records


def load_done_ids(path):
    done = set()
    if not Path(path).exists():
        return done
    for record in read_jsonl(path):
        if record.get("question_id"):
            done.add(record["question_id"])
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Build claim-level citations over generated answers (Phase 9)."
    )
    parser.add_argument(
        "--input",
        default=str(Path(RESULTS_DIR) / GENERATED_FILE),
        help=f"Generated answers JSONL (default: {Path(RESULTS_DIR) / GENERATED_FILE})",
    )
    parser.add_argument(
        "--output",
        default=str(Path(RESULTS_DIR) / OUTPUT_FILE),
        help=f"Citations output JSONL (default: {Path(RESULTS_DIR) / OUTPUT_FILE})",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_ids(out_path)
    records = read_jsonl(in_path)
    if not records:
        sys.exit(f"Error: no records in {in_path}")

    skipped = 0
    built = 0
    failed = 0
    built_at = datetime.now(timezone.utc).isoformat()

    with open(out_path, "a", encoding="utf-8") as out:
        for record in records:
            qid = record["question_id"]
            if qid in done:
                skipped += 1
                continue

            chunks = record.get("retrieved_chunks") or []
            try:
                claims = build_citations(
                    qid,
                    record["question"],
                    record["generated_answer"],
                    chunks,
                    selector=select_supporting_chunks,
                )
            except SourceJudgeError as e:
                failed += 1
                print(f"  ! citation build failed for {qid} [{e.category}]: {e}",
                      file=sys.stderr)
                continue

            valid, errors = validate_citations(claims, chunks)
            entry = {
                "question_id": qid,
                "question": record["question"],
                "answerable": record["answerable"],
                "source_dataset": record.get("source_dataset"),
                "answer": record["generated_answer"],
                "retrieved_chunk_ids": record.get("retrieved_chunk_ids"),
                "claims": claims,
                "citation_version": CITATION_VERSION,
                "validation": {"valid": valid, "errors": errors},
                "built_at": built_at,
            }
            out.write(json.dumps(entry) + "\n")
            out.flush()
            done.add(qid)
            built += 1
            n_citations = sum(len(c["citations"]) for c in claims)
            print(f"  {qid}: {len(claims)} claims, {n_citations} citations")

    print(f"\nRecords in generated file:     {len(records)}")
    print(f"Already built (skipped):       {skipped}")
    print(f"Built this run:                {built}")
    print(f"Failed (left unbuilt):         {failed}")
    print(f"Citations written to:          {out_path}")
    if failed:
        print(
            f"WARNING: {failed} record(s) unbuilt. Re-run to retry (resumable).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
