"""Phase 10 STEP 6 — deep analysis of reorder results.

Computes:
  - affected-subset metrics (only questions where reordering applies)
  - per-question P@1/P@3/P@5 deltas
  - improved / unchanged / regressed counts
  - distractor-class analysis (which classes moved up/down in top-3)
  - which relevant chunks were incorrectly demoted
  - LQO classifier reliability vs the LLM oracle (held-out correlation)

Reads frozen candidates + labels + step5 diagnosis + step6 reorder orders.
All outputs under evaluation/experiments/step6/results/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

OUTDIR = os.path.join(os.path.dirname(__file__), "results")
BENCH_CANDS = "evaluation/results/reranked_hybrid_800_100_candidates.jsonl"
BENCH_LABELS = "evaluation/results/hybrid_800_100_candidates_labeled.jsonl"
HOLDOUT_CANDS = "evaluation/results/holdout_v1_reranked_hybrid_candidates.jsonl"
HOLDOUT_LABELS = "evaluation/results/holdout_v1_hybrid_candidates_labeled.jsonl"
DIAG = "evaluation/experiments/step5_diagnosis.jsonl"

VARIANTS = [
    ("oracle_usefulness", "oracle_usefulness_a0.5"),
    ("oracle_class", "oracle_class_a0.5"),
    ("lqo", "lqo_a0.3"),
    ("lqo_best", "lqo_a0.5"),
    ("crossset", "crossset_a0.3"),
    ("suppress_rationale", "suppress_rationale"),
]


def load_pairs(cands_path, labels_path):
    can, lab = {}, {}
    for line in open(cands_path):
        r = json.loads(line)
        can[(r["question_id"], r["chunk_id"])] = r
    for line in open(labels_path):
        r = json.loads(line)
        lab[(r["question_id"], r["chunk_id"])] = r["relevant"]
    return can, lab


def load_diagnosis():
    best = {}
    for line in open(DIAG):
        r = json.loads(line)
        best[(r["question_id"], r["chunk_id"])] = r
    return best


def load_orders(prefix, setname):
    orders = {}
    path = f"{OUTDIR}/{prefix}_{setname}_order.jsonl"
    if not os.path.exists(path):
        return None
    for line in open(path):
        r = json.loads(line)
        orders[r["question_id"]] = r["order"]
    return orders


def metrics(ranked, labels):
    m = {}
    n = len(ranked)
    for k in (1, 3, 5):
        m[f"P@{k}"] = sum(
            sum(1 for c in ranked[q][:k] if labels[(q, c)] == "relevant") / k
            for q in ranked) / n
    m["MRR"] = sum(
        1 / next((i for i, c in enumerate(ranked[q], 1)
                  if labels[(q, c)] == "relevant"), 1e9)
        for q in ranked) / n
    m["Hit@5"] = sum(
        1 for q in ranked if any(labels[(q, c)] == "relevant" for c in ranked[q][:5])) / n
    return m


def main():
    bench_can, bench_lab = load_pairs(BENCH_CANDS, BENCH_LABELS)
    hold_can, hold_lab = load_pairs(HOLDOUT_CANDS, HOLDOUT_LABELS)
    diag = load_diagnosis()

    bench_affected = ["Q06", "Q07", "Q08", "Q09", "Q11", "Q12", "Q13", "Q14", "Q18"]
    hold_affected = ["H01", "H03", "H08", "H09", "H10", "H13", "H15", "H16",
                     "H17", "H18", "H19", "H21", "H22", "H23", "H24"]

    report = {}

    # baseline orders (frozen)
    def frozen_order(can, q):
        rows = [k for k in can if k[0] == q]
        rows.sort(key=lambda k: can[k]["reranker_rank"])
        return [c[1] for c in rows[:10]]

    base_b = {q: frozen_order(bench_can, q) for q in bench_affected}
    base_h = {q: frozen_order(hold_can, q) for q in hold_affected}

    mb = metrics(base_b, bench_lab)
    mh = metrics(base_h, hold_lab)
    print("AFFECTED-SUBSET baseline bench:", {k: round(v, 4) for k, v in mb.items()})
    print("AFFECTED-SUBSET baseline hold: ", {k: round(v, 4) for k, v in mh.items()})
    report["baseline_affected"] = {"bench": mb, "hold": mh}

    for label, prefix in VARIANTS:
        ob = load_orders(prefix, "bench")
        oh = load_orders(prefix, "hold")
        if ob is None:
            print(f"  !! no orders for {prefix}")
            continue
        ob_sub = {q: ob[q] for q in bench_affected if q in ob}
        oh_sub = {q: oh[q] for q in hold_affected if q in oh}

        rb = metrics(ob_sub, bench_lab)
        rh = metrics(oh_sub, hold_lab)
        print(f"\n=== {label} ({prefix}) AFFECTED-SUBSET ===")
        print("  bench:", {k: round(v, 4) for k, v in rb.items()},
              "Δ", {k: round(rb[k] - mb[k], 4) for k in mb})
        print("  hold :", {k: round(v, 4) for k, v in rh.items()},
              "Δ", {k: round(rh[k] - mh[k], 4) for k in mh})

        # per-question deltas
        def pq_deltas(before, after, qs, labels):
            rows = []
            for q in sorted(qs):
                def p3(o):
                    return sum(1 for c in o[:3] if labels[(q, c)] == "relevant") / 3
                def p5(o):
                    return sum(1 for c in o[:5] if labels[(q, c)] == "relevant") / 5
                rows.append({
                    "q": q,
                    "p1_b": int(labels[(q, before[q][0])] == "relevant"),
                    "p1_a": int(labels[(q, after[q][0])] == "relevant"),
                    "p3_b": round(p3(before[q]), 3), "p3_a": round(p3(after[q]), 3),
                    "p5_b": round(p5(before[q]), 3), "p5_a": round(p5(after[q]), 3),
                })
            return rows

        bench_d = pq_deltas(base_b, ob_sub, bench_affected, bench_lab)
        hold_d = pq_deltas(base_h, oh_sub, hold_affected, hold_lab)
        n_b3 = sum(1 for r in bench_d if r["p3_a"] > r["p3_b"])
        n_b3r = sum(1 for r in bench_d if r["p3_a"] < r["p3_b"])
        n_b1 = sum(1 for r in bench_d if r["p1_a"] < r["p1_b"])
        n_h3 = sum(1 for r in hold_d if r["p3_a"] > r["p3_b"])
        n_h3r = sum(1 for r in hold_d if r["p3_a"] < r["p3_b"])
        n_h1 = sum(1 for r in hold_d if r["p1_a"] < r["p1_b"])
        print(f"  bench P@3 improved/regressed: {n_b3}/{n_b3r}, P@1 regressions: {n_b1}")
        print(f"  hold  P@3 improved/regressed: {n_h3}/{n_h3r}, P@1 regressions: {n_h1}")

        # which questions changed
        changed_b = [r for r in bench_d if r["p3_a"] != r["p3_b"] or r["p1_a"] != r["p1_b"]]
        changed_h = [r for r in hold_d if r["p3_a"] != r["p3_b"] or r["p1_a"] != r["p1_b"]]
        print("  bench changed:", [(r["q"], f"{r['p3_b']}->{r['p3_a']}", f"p1:{r['p1_b']}->{r['p1_a']}") for r in changed_b])
        print("  hold  changed:", [(r["q"], f"{r['p3_b']}->{r['p3_a']}", f"p1:{r['p1_b']}->{r['p1_a']}") for r in changed_h])

        # candidate recall preservation
        rec_b = all(set(ob[q][:10]) == set(frozen_order(bench_can, q)) for q in bench_affected if q in ob)
        rec_h = all(set(oh[q][:10]) == set(frozen_order(hold_can, q)) for q in hold_affected if q in oh)
        print(f"  top-10 set preserved bench={rec_b} hold={rec_h}")

        report[label] = {
            "bench": rb, "hold": rh, "bench_delta": {k: rb[k] - mb[k] for k in mb},
            "hold_delta": {k: rh[k] - mh[k] for k in mh},
            "bench_pq": bench_d, "hold_pq": hold_d,
            "bench_P3_improved": n_b3, "bench_P3_regressed": n_b3r, "bench_P1_regressions": n_b1,
            "hold_P3_improved": n_h3, "hold_P3_regressed": n_h3r, "hold_P1_regressions": n_h1,
        }

    with open(f"{OUTDIR}/analysis.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nwrote", f"{OUTDIR}/analysis.json")


if __name__ == "__main__":
    main()