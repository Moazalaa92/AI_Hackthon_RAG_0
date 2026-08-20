"""Calibrate the evidence-support bands of src/rating.py against judged answers.

Offline only: every input is a frozen artifact and NO LLM call is made. The
question this answers is narrow and it is the only claim the UI is allowed to
make about a band:

    for answers that fall in band B, what fraction were judged
    "correct" by the Phase 8.5 correctness judge?

Inputs (all read-only):
  evaluation/results/generation_v2_judged.jsonl   correctness / grounding labels
  evaluation/results/citations_v1.jsonl           claims, citations, validation
  evaluation/results/reranked_hybrid_800_100_top10.jsonl      benchmark Top-10
  evaluation/results/holdout_v1_reranked_hybrid_top10.jsonl   holdout Top-10
  evaluation/results/hybrid_800_100_candidates.jsonl          retrieved_by
  evaluation/results/holdout_v1_hybrid_candidates.jsonl       retrieved_by

The band assignment itself is NOT reimplemented here: this script builds the
same `Answer` / `SafetyResult` inputs the API builds and calls
`src.rating.rate_answer`, so a rule change cannot silently invalidate the
published table.

Usage:
    python scripts/calibrate_rating_bands.py                  # current constants
    python scripts/calibrate_rating_bands.py --sweep          # threshold sweep
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import Answer
from src.rating import T_MARGIN, T_TOP1, rate_answer
from src.safety import classify_evidence

JUDGED = "evaluation/results/generation_v2_judged.jsonl"
CITATIONS = "evaluation/results/citations_v1.jsonl"
TOP10 = (
    "evaluation/results/reranked_hybrid_800_100_top10.jsonl",
    "evaluation/results/holdout_v1_reranked_hybrid_top10.jsonl",
)
CANDIDATES = (
    "evaluation/results/hybrid_800_100_candidates.jsonl",
    "evaluation/results/holdout_v1_hybrid_candidates.jsonl",
)
BANDS = ("high", "medium", "low", "refused")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _wilson(successes, total):
    """95% Wilson score interval; (None, None) for an empty band."""
    if not total:
        return None, None
    z = 1.959963984540054
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _load_chunks():
    """question_id -> Top-10 chunk dicts in rank order, with retrieved_by joined."""
    retrieved_by = {}
    for path in CANDIDATES:
        for record in _read_jsonl(path):
            retrieved_by[(record["question_id"], record["chunk_id"])] = record.get(
                "retrieved_by"
            )
    chunks = {}
    for path in TOP10:
        for record in _read_jsonl(path):
            qid = record["question_id"]
            chunks.setdefault(qid, []).append(
                {
                    "chunk_id": record["chunk_id"],
                    "document_id": record.get("document_id"),
                    "page": record.get("page"),
                    "page_label": record.get("page_label"),
                    "reranker_rank": record.get("rank"),
                    "reranker_score": record.get("score"),
                    "retrieved_by": retrieved_by.get((qid, record["chunk_id"])),
                    "chunk_text": record.get("chunk_text", ""),
                }
            )
    for records in chunks.values():
        records.sort(key=lambda item: item["reranker_rank"])
    return chunks


def build_cases():
    """One case per judged question: the rating inputs plus the judged label."""
    judged = {record["question_id"]: record for record in _read_jsonl(JUDGED)}
    citations = {record["question_id"]: record for record in _read_jsonl(CITATIONS)}
    chunks = _load_chunks()

    missing = sorted(set(judged) - set(citations)) + sorted(set(judged) - set(chunks))
    if missing:
        raise SystemExit(f"missing artifacts for question ids: {missing}")

    cases = []
    for qid, judged_record in sorted(judged.items()):
        citation_record = citations[qid]
        validation = citation_record.get("validation") or {}
        claims = citation_record.get("claims") or []
        answer = Answer(
            question=citation_record["question"],
            answer=citation_record.get("answer", ""),
            claims=claims,
            retrieved_chunks=chunks[qid],
            citations_valid=validation.get("valid"),
            validation_errors=validation.get("errors") or [],
        )
        cases.append(
            {
                "question_id": qid,
                "answer": answer,
                "safety": classify_evidence(answer.answer, claims),
                "answerable": bool(judged_record.get("answerable")),
                "correctness": judged_record.get("correctness"),
                "grounding": judged_record.get("grounding"),
                "abstention": judged_record.get("abstention"),
            }
        )
    return cases


def band_table(cases, top1_threshold, margin_threshold):
    """band -> counts of judged correctness labels for the answerable cases."""
    table = {
        band: {
            "n": 0,
            "correct": 0,
            "partially_correct": 0,
            "incorrect": 0,
            "unlabeled": 0,
            "negatives": 0,
            "question_ids": [],
        }
        for band in BANDS
    }
    for case in cases:
        rating = rate_answer(
            case["answer"],
            case["safety"],
            top1_threshold=top1_threshold,
            margin_threshold=margin_threshold,
        )
        row = table[rating.band]
        row["question_ids"].append(case["question_id"])
        if not case["answerable"]:
            row["negatives"] += 1
            continue
        row["n"] += 1
        label = case["correctness"]
        row[label if label in row else "unlabeled"] += 1
    return table


def _print_table(table, top1_threshold, margin_threshold):
    print(f"\nT_TOP1={top1_threshold}  T_MARGIN={margin_threshold}")
    header = f"{'band':<9}{'answerable':>11}{'correct':>9}{'partial':>9}{'incorrect':>10}{'correct rate':>14}{'95% CI':>18}{'negatives':>11}"
    print(header)
    print("-" * len(header))
    for band in BANDS:
        row = table[band]
        rate = row["correct"] / row["n"] if row["n"] else None
        low, high = _wilson(row["correct"], row["n"])
        rate_text = f"{rate:.3f}" if rate is not None else "-"
        ci_text = f"[{low:.3f}, {high:.3f}]" if low is not None else "-"
        print(
            f"{band:<9}{row['n']:>11}{row['correct']:>9}{row['partially_correct']:>9}"
            f"{row['incorrect']:>10}{rate_text:>14}{ci_text:>18}{row['negatives']:>11}"
        )
    for band in BANDS:
        print(f"  {band}: {', '.join(table[band]['question_ids']) or '(none)'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="sweep both thresholds")
    parser.add_argument("--json-out", help="write the sweep/table as JSON")
    args = parser.parse_args()

    cases = build_cases()
    answerable = sum(case["answerable"] for case in cases)
    correct = sum(
        case["answerable"] and case["correctness"] == "correct" for case in cases
    )
    print(
        f"{len(cases)} judged questions | {answerable} answerable | "
        f"baseline correct rate {correct}/{answerable} = {correct / answerable:.3f}"
    )

    results = []
    if args.sweep:
        top1_scores = sorted(
            case["answer"].retrieved_chunks[0]["reranker_score"] for case in cases
        )
        grid_top1 = [-6.0, -4.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        grid_margin = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
        print(
            f"top-1 reranker score range: {top1_scores[0]:.3f} .. {top1_scores[-1]:.3f}"
            f" (median {top1_scores[len(top1_scores) // 2]:.3f})"
        )
        for t1 in grid_top1:
            for tm in grid_margin:
                table = band_table(cases, t1, tm)
                results.append(
                    {"T_TOP1": t1, "T_MARGIN": tm, "table": table}
                )
                _print_table(table, t1, tm)
    else:
        table = band_table(cases, T_TOP1, T_MARGIN)
        results.append({"T_TOP1": T_TOP1, "T_MARGIN": T_MARGIN, "table": table})
        _print_table(table, T_TOP1, T_MARGIN)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
