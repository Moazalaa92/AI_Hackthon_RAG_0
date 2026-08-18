"""CLI: rank-movement analysis for the cross-encoder reranking experiment.

Reads ONLY three frozen artifacts:
  - hybrid_800_100_candidates.jsonl            (fusion rank 1..n per question)
  - hybrid_800_100_candidates_labeled.jsonl    (LLM relevance per question/chunk)
  - reranked_hybrid_800_100_candidates.jsonl   (reranker_rank 1..n per question)

Joins on (question_id, document_id). For every RELEVANT chunk in the candidate
pool it compares the hybrid (RRF fusion) rank against the reranked rank and
classifies the movement as improved / kept (within ±1) / worsened. This keeps
candidate recall (the pool) separate from ranking quality (the reranker): it
only evaluates chunks that were ALREADY in the input candidate pool.

No LLM, no retrieval, no reranker. Never modifies the input files.

Usage:
    python scripts/rerank_analysis.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR, RESULTS_DIR

NAME = "reranked_hybrid_800_100"


def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    fusion = {}
    for r in read_jsonl(Path(RESULTS_DIR) / "hybrid_800_100_candidates.jsonl"):
        fusion[(r["question_id"], r["document_id"])] = r["rank"]

    labels = {}
    for r in read_jsonl(Path(RESULTS_DIR) / "hybrid_800_100_candidates_labeled.jsonl"):
        labels[(r["question_id"], r["document_id"])] = r["relevant"]

    reranked = {}
    for r in read_jsonl(Path(RESULTS_DIR) / f"{NAME}_candidates.jsonl"):
        reranked[(r["question_id"], r["document_id"])] = r["reranker_rank"]

    by_question = {}
    for (qid, doc_id), relevant in labels.items():
        if relevant != "relevant":
            continue
        if (qid, doc_id) not in fusion:
            continue
        hyb = fusion[(qid, doc_id)]
        rr = reranked[(qid, doc_id)]
        movement = (
            "improved"
            if rr < hyb
            else "worsened"
            if rr > hyb
            else "kept"
        )
        by_question.setdefault(qid, []).append(
            {"document_id": doc_id, "fusion_rank": hyb, "reranked_rank": rr, "movement": movement}
        )

    print("Rank movement of every pool-relevant chunk (fusion rank -> reranked rank):")
    print(f"{'Q':>4} {'chunk':>10} {'fusion':>7} {'reranked':>9}  {'movement':>9}")
    for qid in sorted(by_question):
        for c in sorted(by_question[qid], key=lambda x: x["fusion_rank"]):
            print(
                f"{qid:>4} {c['document_id']:>10} {c['fusion_rank']:>7} "
                f"{c['reranked_rank']:>9}  {c['movement']:>9}"
            )

    agg = {"improved": 0, "kept": 0, "worsened": 0}
    improved_into_top3 = 0
    for qid in by_question:
        for c in by_question[qid]:
            agg[c["movement"]] += 1
            if c["reranked_rank"] <= 3:
                improved_into_top3 += 1

    total = sum(agg.values())
    print(f"\nAggregates ({total} pool-relevant chunks):")
    print(f"  improved : {agg['improved']}")
    print(f"  kept     : {agg['kept']} (rank unchanged or within ±0)")
    print(f"  worsened : {agg['worsened']}")
    print(f"  moved into Top-3 by reranker: {improved_into_top3}")

    out_path = Path(METRICS_DIR) / f"{NAME}_rank_movement.json"
    payload = {
        "experiment": NAME,
        "per_question": by_question,
        "aggregates": agg,
        "relevant_chunks_in_top3_after_rerank": improved_into_top3,
        "caveat": (
            "Relevant set = LLM-judged labels from the hybrid candidate pool "
            "(unchanged judge). Ranks compared: hybrid RRF fusion rank vs "
            "cross-encoder reranked rank, for chunks already in the pool."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()