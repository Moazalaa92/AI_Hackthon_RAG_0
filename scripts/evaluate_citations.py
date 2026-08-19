"""CLI: evaluate citation coverage, support and traceability (Phase 9).

Reads the citations built by scripts/run_citations.py and measures:

  A. Citation Coverage   : do claims have a citation?
       coverage = claims with >=1 citation / all claims
       (deliberate-negative refusals are excluded: they are refusal claims and
        get no citations by design)
  B. Citation Support    : does the cited chunk actually support the claim?
       per-citation label from the LLM evaluator
       (supported / partially_supported / unsupported)
  C. Citation Traceability: deterministic - does every citation point to a real
       retrieved chunk with matching metadata and exact text?
       validated at build time by src.sources.validate_citations and carried
       forward here, plus a per-citation check that the cited chunk_id is in
       the question's retrieved set.

The LLM (src.sources_judge.judge_citation_support) is used ONLY as an
evaluator/judge - never as ground truth. Traceability and the claim/citation
counts are deterministic.

Outputs:
    evaluation/results/citations_v1_judged.jsonl   (per-question judged results)
    evaluation/metrics/citations_evaluation.json    (aggregate metrics)
    prints a summary table to stdout

Resumable: questions already present in the judged output are skipped.

Usage:
    python scripts/evaluate_citations.py
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR, RESULTS_DIR
from src.sources import is_refusal_claim
from src.sources_judge import SourceJudgeError, judge_citation_support

CITATIONS_FILE = "citations_v1.jsonl"
JUDGED_FILE = "citations_v1_judged.jsonl"
METRICS_FILE = "citations_evaluation.json"


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
    """Question IDs already fully judged (all citations have a support label).

    A question with any citation whose support is still None/error is NOT
    considered done, so a re-run retries it (resumable at citation level).
    """
    done = set()
    if not Path(path).exists():
        return done
    for record in read_jsonl(path):
        qid = record.get("question_id")
        if not qid:
            continue
        all_judged = all(
            cit.get("support") is not None
            for claim in record.get("claims", [])
            for cit in claim.get("citations", [])
        )
        if all_judged:
            done.add(qid)
    return done


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate citation coverage, support and traceability (Phase 9)."
    )
    parser.add_argument(
        "--input",
        default=str(Path(RESULTS_DIR) / CITATIONS_FILE),
        help=f"Citations JSONL (default: {Path(RESULTS_DIR) / CITATIONS_FILE})",
    )
    parser.add_argument(
        "--output",
        default=str(Path(RESULTS_DIR) / JUDGED_FILE),
        help=f"Judged output JSONL (default: {Path(RESULTS_DIR) / JUDGED_FILE})",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(in_path)
    if not records:
        sys.exit(f"Error: no records in {in_path}")

    done = load_done_ids(out_path)

    # Keep already-judged records in memory so re-judged (previously partial)
    # questions overwrite their old line instead of appending a duplicate.
    existing = {}
    for record in read_jsonl(out_path) if out_path.exists() else []:
        existing[record["question_id"]] = record
    if done:
        print(f"Already fully judged (skipped): {len(done)}")

    judged_at = datetime.now(timezone.utc).isoformat()

    skipped = 0
    judged = 0
    failed = 0

    for record in records:
        qid = record["question_id"]
        if qid in done:
            skipped += 1
            continue

        claims = []
        for claim in record["claims"]:
            entry = {
                "claim_id": claim["claim_id"],
                "claim_text": claim["claim_text"],
                "citations": [],
            }
            for citation in claim["citations"]:
                cit_entry = dict(citation)
                try:
                    support = judge_citation_support(
                        record["question"],
                        claim["claim_text"],
                        citation["supporting_text"],
                    )
                except SourceJudgeError as e:
                    failed += 1
                    print(f"  ! support judge failed for {qid} {claim['claim_id']} "
                          f"[{e.category}]: {e}", file=sys.stderr)
                    cit_entry["support"] = None
                    cit_entry["support_error"] = str(e)
                else:
                    cit_entry["support"] = support["label"]
                    cit_entry["support_reason"] = support["reason"]
                    cit_entry["support_judge"] = support.get("judge", "llm")
                entry["citations"].append(cit_entry)
            claims.append(entry)

        # Traceability comes from the build step (run_citations validated every
        # citation against the actual retrieved chunks). We carry it forward;
        # the judged JSONL only adds the support labels.
        valid = bool(record.get("validation", {}).get("valid"))
        errors = record.get("validation", {}).get("errors") or []
        out_record = {
            "question_id": qid,
            "question": record["question"],
            "answerable": record["answerable"],
            "retrieved_chunk_ids": record.get("retrieved_chunk_ids"),
            "claims": claims,
            "validation": {"valid": valid, "errors": errors},
            "judge": "llm",
            "judged_at": judged_at,
        }
        existing[qid] = out_record
        judged += 1

    # Rewrite the full judged file from the merged dict (deduplicates re-judged
    # questions and keeps the file consistent even after a partial run).
    ordered = [existing[r["question_id"]] for r in records if r["question_id"] in existing]
    with open(out_path, "w", encoding="utf-8") as out:
        for record in ordered:
            out.write(json.dumps(record) + "\n")

    print(f"Records in citations file:     {len(records)}")
    print(f"Already judged (skipped):      {skipped}")
    print(f"Judged this run:               {judged}")
    print(f"Failed (left unjudged):        {failed}")
    print(f"Judged results written to:     {out_path}")

    # ---- aggregate metrics over the full judged file ----
    judged_records = []
    for record in read_jsonl(out_path):
        if record.get("claims") is not None:
            judged_records.append(record)

    metrics = aggregate(judged_records)
    write_json(Path(METRICS_DIR) / METRICS_FILE, metrics)
    print_metrics(metrics)
    print(f"Metrics written to:            {Path(METRICS_DIR) / METRICS_FILE}")


def aggregate(records):
    """Compute coverage / support / traceability over all judged records."""
    total_claims = 0
    no_evidence_claims = 0
    answerable_claims = 0
    claims_with_citation = 0
    citations_total = 0
    citations_supported = Counter()
    trace_valid = 0
    trace_errors = 0
    questions_with_validation_error = 0

    for record in records:
        if record.get("validation", {}).get("valid") is not True:
            questions_with_validation_error += 1
            trace_errors += 1
        for claim in record["claims"]:
            total_claims += 1
            n_citations = len(claim["citations"])
            citations_total += n_citations
            if n_citations > 0:
                claims_with_citation += 1
            if record["answerable"]:
                if is_refusal_claim(claim["claim_text"]):
                    no_evidence_claims += 1
                else:
                    answerable_claims += 1
            for citation in claim["citations"]:
                label = citation.get("support")
                if label in ("supported", "partially_supported", "unsupported"):
                    citations_supported[label] += 1
        if record.get("validation", {}).get("valid") is True:
            trace_valid += 1

    # Traceability per-citation: recompute deterministically.
    # validate_citations already ran per question; count per-citation validity.
    per_citation_valid = 0
    per_citation_total = 0
    for record in records:
        chunks = []
        # we do not re-store chunks in judged records; reuse input file chunks
        # via the validation flag computed at build time is not per-citation, so
        # count per-citation from the citation metadata + retrieved ids stored.
        retrieved = set(record.get("retrieved_chunk_ids") or [])
        for claim in record["claims"]:
            for citation in claim["citations"]:
                per_citation_total += 1
                if citation.get("chunk_id") in retrieved:
                    per_citation_valid += 1

    coverage = claims_with_citation / answerable_claims if answerable_claims else 0.0
    return {
        "n_questions": len(records),
        "n_claims": total_claims,
        "n_answerable_claims": answerable_claims,
        "n_no_evidence_claims": no_evidence_claims,
        "n_claims_with_citation": claims_with_citation,
        "coverage": round(coverage, 4),
        "n_citations": citations_total,
        "citations_by_support": dict(citations_supported),
        "support_total_judged": sum(citations_supported.values()),
        "traceability_questions_valid": trace_valid,
        "n_questions": len(records),
        "questions_with_validation_error": questions_with_validation_error,
        "traceability_citations_in_retrieved": per_citation_valid,
        "traceability_citations_total": per_citation_total,
    }


def print_metrics(m):
    print("\n===== Citation evaluation =====")
    print(f"Questions:                     {m['n_questions']}")
    print(f"Claims (all):                  {m['n_claims']}")
    print(f"Answerable claims (no-evidence excluded): {m['n_answerable_claims']}")
    print(f"No-evidence claims (uncited by design):   {m['n_no_evidence_claims']}")
    print(f"Claims with >=1 citation:      {m['n_claims_with_citation']}")
    print(f"Coverage (answerable):         {m['coverage']:.1%}")
    print(f"Citations total:               {m['n_citations']}")
    total = m["support_total_judged"]
    if total:
        for label in ("supported", "partially_supported", "unsupported"):
            count = m["citations_by_support"].get(label, 0)
            print(f"  {label:<22}{count:>5} ({count / total:.1%})")
    print(f"Traceability (questions valid): {m['traceability_questions_valid']}/{m['n_questions']}")
    print(f"Traceability (citations point to retrieved chunk): "
          f"{m['traceability_citations_in_retrieved']}/{m['traceability_citations_total']}")


if __name__ == "__main__":
    main()
