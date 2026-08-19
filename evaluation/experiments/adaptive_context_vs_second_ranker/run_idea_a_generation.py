"""Idea A — Adaptive Context Depth, targeted Generation experiment (OFFLINE).

Instead of always passing the frozen Top-10 to Generation, pass all reranked
candidate chunks whose per-question min-max normalized reranker score is above
a threshold. This can TRIM the context (drop low-confidence chunks) and, more
interestingly, can EXPAND it (include high-confidence chunks ranked 11+ that
the frozen Top-10 never shows Generation, e.g. Q11 r13 norm=0.83, H21 r11).

Methodology mirrors scripts/run_generation_evaluation.py + evaluate_generation.py
(leakage control and judge calls identical):
    correctness  : question + reference + generated answer   (NO context)
    grounding    : question + context + generated answer     (NO reference)
    completeness : question + reference + context + answer

Scope (targeted first, per user instruction — do NOT waste API tokens):
    bench difficult : Q06 Q07 Q08 Q09 Q11 Q12 Q13 Q14 Q18
    holdout sampled : H01 H08 H21 H23
    thresholds      : one threshold per run (--threshold)

Generation uses the frozen src/generation.generate_answer unchanged.

Output (per threshold):
    results/ideaA_t{thr}_{set}_generated.jsonl
    results/ideaA_t{thr}_{set}_judged.jsonl

Usage:
    python run_idea_a_generation.py --threshold 0.7 --stage generate
    python run_idea_a_generation.py --threshold 0.7 --stage judge
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import RESULTS_DIR
from src.generation import build_context, generate_answer
from src.generation_judge import (
    JudgeError,
    judge_completeness,
    judge_correctness,
    judge_grounding,
)

OUTDIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTDIR, exist_ok=True)

BENCH_CANDS = "evaluation/results/reranked_hybrid_800_100_candidates.jsonl"
BENCH_LABELS = "evaluation/results/hybrid_800_100_candidates_labeled.jsonl"
HOLDOUT_CANDS = "evaluation/results/holdout_v1_reranked_hybrid_candidates.jsonl"
HOLDOUT_LABELS = "evaluation/results/holdout_v1_hybrid_candidates_labeled.jsonl"
DATASET_FILE = "evaluation/generation_dataset_v1.json"

BENCH_TARGETS = ["Q06", "Q07", "Q08", "Q09", "Q11", "Q12", "Q13", "Q14", "Q18"]
HOLDOUT_TARGETS = ["H01", "H08", "H21", "H23"]

# Number of chunks per question in the frozen candidate pool (upper bound guard)
MAX_CONTEXT = 25


def load_jsonl(path):
    recs = []
    for line in open(path):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def load_dataset():
    with open(DATASET_FILE) as f:
        return json.load(f)


def load_pairs(cands_path, labels_path):
    can, lab = {}, {}
    for line in open(cands_path):
        r = json.loads(line)
        can[(r["question_id"], r["chunk_id"])] = r
    for line in open(labels_path):
        r = json.loads(line)
        lab[(r["question_id"], r["chunk_id"])] = r["relevant"]
    return can, lab


def select_adaptive_chunks(qid, can_by_q, threshold):
    """Return candidate chunks for `qid` with min-max normalized score >= threshold.

    Normalization is per-question over the FULL candidate pool (not the top-10)
    so the threshold has a stable meaning regardless of the pool's score spread.
    Chunks are ordered by reranker_rank (ties broken by rank).
    """
    rows = sorted(can_by_q.get(qid, []), key=lambda r: r["reranker_rank"])
    if not rows:
        return []
    scores = [r["reranker_score"] for r in rows]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return rows[:10]
    selected = [
        r for r in rows
        if (r["reranker_score"] - lo) / (hi - lo) >= threshold
    ]
    # never produce an empty context; keep at least the top-1
    if not selected:
        selected = rows[:1]
    return selected[:MAX_CONTEXT]


def build_question_map(dataset):
    return {q["id"]: q for q in dataset["questions"]}


def generate_stage(threshold):
    dataset = load_dataset()
    qmap = build_question_map(dataset)

    bench_can, _ = load_pairs(BENCH_CANDS, BENCH_LABELS)
    hold_can, _ = load_pairs(HOLDOUT_CANDS, HOLDOUT_LABELS)

    def pool(can):
        by_q = defaultdict(list)
        for (q, _c), r in can.items():
            by_q[q].append(r)
        return by_q

    bench_pool = pool(bench_can)
    hold_pool = pool(hold_can)

    targets = [("bench", BENCH_TARGETS, bench_pool),
               ("hold", HOLDOUT_TARGETS, hold_pool)]

    generated_at = datetime.now(timezone.utc).isoformat()
    total = 0
    for setname, qids, pool_by_q in targets:
        out_path = os.path.join(OUTDIR, f"ideaA_t{threshold:.2f}_{setname}_generated.jsonl")
        done = set()
        if os.path.exists(out_path):
            for r in load_jsonl(out_path):
                done.add(r["question_id"])

        with open(out_path, "a") as out:
            for qid in qids:
                if qid in done:
                    continue
                q = qmap[qid]
                chunks = select_adaptive_chunks(qid, pool_by_q, threshold)
                if not chunks:
                    print(f"  ! no candidate chunks for {qid}")
                    continue
                answer = generate_answer(q["question"], chunks, max_tokens=600)
                rec = {
                    "question_id": qid,
                    "question": q["question"],
                    "answerable": q["answerable"],
                    "reference": q.get("reference"),
                    "threshold": threshold,
                    "n_chunks": len(chunks),
                    "generated_answer": answer,
                    "generated_at": generated_at,
                    "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
                    "retrieved_chunks": chunks,
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                done.add(qid)
                total += 1
                print(f"  {setname} {qid}: {len(chunks)} chunks, {len(answer)} chars")
    print(f"\nGenerated {total} answers (resumable).")


def judge_stage(threshold):
    dataset = load_dataset()
    qmap = build_question_map(dataset)
    bench_can, bench_lab = load_pairs(BENCH_CANDS, BENCH_LABELS)
    hold_can, hold_lab = load_pairs(HOLDOUT_CANDS, HOLDOUT_LABELS)

    def relevant_map(lab):
        out = defaultdict(set)
        for (q, c), rel in lab.items():
            if rel == "relevant":
                out[q].add(c)
        return out

    bench_rel = relevant_map(bench_lab)
    hold_rel = relevant_map(hold_lab)
    relevant_by_q = {**bench_rel, **hold_rel}

    judged_at = datetime.now(timezone.utc).isoformat()
    total = 0
    for setname, _qids in [("bench", BENCH_TARGETS), ("hold", HOLDOUT_TARGETS)]:
        in_path = os.path.join(OUTDIR, f"ideaA_t{threshold:.2f}_{setname}_generated.jsonl")
        out_path = os.path.join(OUTDIR, f"ideaA_t{threshold:.2f}_{setname}_judged.jsonl")
        if not os.path.exists(in_path):
            print(f"  ! missing {in_path}")
            continue
        done = set()
        if os.path.exists(out_path):
            for r in load_jsonl(out_path):
                done.add(r["question_id"])

        with open(out_path, "a") as out:
            for rec in load_jsonl(in_path):
                qid = rec["question_id"]
                if qid in done:
                    continue
                reference = rec.get("reference")
                answer = rec["generated_answer"]
                context = build_context(rec["retrieved_chunks"])

                entry = {
                    "question_id": qid,
                    "question": rec["question"],
                    "answerable": rec["answerable"],
                    "threshold": threshold,
                    "n_chunks": rec["n_chunks"],
                    "generated_answer": answer,
                    "retrieved_chunk_ids": rec["retrieved_chunk_ids"],
                }

                try:
                    if rec["answerable"]:
                        correctness = judge_correctness(
                            rec["question"], reference, answer)
                        grounding = judge_grounding(
                            rec["question"], context, answer)
                        completeness = judge_completeness(
                            rec["question"], reference, context, answer)
                        entry["correctness"] = correctness["label"]
                        entry["grounding"] = grounding["label"]
                        entry["completeness"] = completeness["label"]
                        entry["abstention"] = "not_applicable"
                        entry["judge_reasons"] = {
                            "correctness": correctness["reason"],
                            "grounding": grounding["reason"],
                            "completeness": completeness["reason"],
                        }
                    else:
                        entry["correctness"] = "not_applicable"
                        entry["grounding"] = "not_applicable"
                        entry["completeness"] = "not_applicable"
                        entry["abstention"] = "not_applicable"
                        entry["judge_reasons"] = {}
                except JudgeError as e:
                    print(f"  ! judge failed for {qid} [{e.category}]: {e}")
                    continue

                if rec["answerable"] and entry["correctness"] != "correct":
                    rel = relevant_by_q.get(qid, set())
                    present = any(c in rel for c in rec["retrieved_chunk_ids"])
                    entry["failure_type"] = (
                        "generation_failure" if present else "retrieval_failure")
                else:
                    entry["failure_type"] = None
                entry["evidence_present"] = (
                    any(c in relevant_by_q.get(qid, set())
                        for c in rec["retrieved_chunk_ids"])
                    if rec["answerable"] else None)
                entry["judge"] = "llm"
                entry["judged_at"] = judged_at

                out.write(json.dumps(entry) + "\n")
                out.flush()
                done.add(qid)
                total += 1
    print(f"\nJudged {total} answers (resumable).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--stage", choices=["generate", "judge"], required=True)
    args = ap.parse_args()
    if args.stage == "generate":
        generate_stage(args.threshold)
    else:
        judge_stage(args.threshold)


if __name__ == "__main__":
    main()
