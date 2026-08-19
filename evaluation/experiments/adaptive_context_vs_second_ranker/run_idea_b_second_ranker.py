"""Idea B — Second Lightweight Ranking Layer (OFFLINE, ZERO LLM calls).

Deterministic second-pass reorder of the frozen reranked Top-10, using a
weighted combination of the normalized cross-encoder score and a deterministic
secondary signal:

    combined = (1 - alpha) * norm_ce + alpha * norm(secondary)

Secondary signals (all deterministic, available in the frozen candidate files
or computable from question + chunk text):
    bm25_rank    inverse bm25 rank (1/rank), 0 if missing
    bm25_score   per-question min-max of bm25 score (0 if missing)
    dense_rank   inverse dense rank (1/rank), 0 if missing
    dense_score  per-question min-max of dense score (0 if missing)
    fusion_rank  inverse RRF fusion rank (1/rank)
    lex_overlap  question/chunk lexical Jaccard overlap
    key_coverage fraction of question key terms covered by the chunk
    scope_overlap coverage of scope/qualifier terms (question minus key terms)

alpha sweep: {0.0, 0.1, 0.2, 0.3, 0.5}  (0.0 == frozen order sanity check)

Scope of reordering: the frozen Top-10 pool only (same as STEP 6). Never pulls
in chunks from beyond top-10, never modifies the frozen pipeline.

Evaluation: per-question P@1/P@3/P@5/MRR/Hit@5 against the frozen binary
relevance labels, for benchmark (Q01-Q20) and holdout (H01-H27) separately.
Per-question deltas + regression counts are also reported.

All outputs under evaluation/experiments/adaptive_context_vs_second_ranker/results/
"""

import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

OUTDIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTDIR, exist_ok=True)

BENCH_CANDS = "evaluation/results/reranked_hybrid_800_100_candidates.jsonl"
BENCH_LABELS = "evaluation/results/hybrid_800_100_candidates_labeled.jsonl"
HOLDOUT_CANDS = "evaluation/results/holdout_v1_reranked_hybrid_candidates.jsonl"
HOLDOUT_LABELS = "evaluation/results/holdout_v1_hybrid_candidates_labeled.jsonl"

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5]
SECONDARY_SIGNALS = [
    "bm25_rank",
    "bm25_score",
    "dense_rank",
    "dense_score",
    "fusion_rank",
    "lex_overlap",
    "key_coverage",
    "scope_overlap",
]

STOPWORDS = set(
    "a an the of in on for and or with to from by as at is are be was were "
    "may should this that these those have has had do does did not no nor but "
    "can could would should if than then what when where which who whom whose "
    "all any both each few more most other some such only own same so too very "
    "s just about into over under again further once here there why how".split()
)


def load_pairs(cands_path, labels_path):
    can, lab = {}, {}
    for line in open(cands_path):
        r = json.loads(line)
        can[(r["question_id"], r["chunk_id"])] = r
    for line in open(labels_path):
        r = json.loads(line)
        lab[(r["question_id"], r["chunk_id"])] = r["relevant"]
    return can, lab


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def key_terms(question_text):
    toks = tokenize(question_text)
    return toks - STOPWORDS


def secondary_features(cand, question_text):
    """Compute deterministic secondary signals for one candidate chunk."""
    def inv(r):
        return 1.0 / r if isinstance(r, (int, float)) and r and r > 0 else 0.0

    qt = tokenize(question_text)
    ct = tokenize(cand.get("chunk_text", ""))
    kt = key_terms(question_text)
    scope = qt - kt

    union = qt | ct
    inter = qt & ct

    return {
        "bm25_rank": inv(cand.get("bm25_rank")),
        "bm25_score": float(cand["bm25_score"]) if isinstance(
            cand.get("bm25_score"), (int, float)) else 0.0,
        "dense_rank": inv(cand.get("dense_rank")),
        "dense_score": float(cand["dense_score"]) if isinstance(
            cand.get("dense_score"), (int, float)) else 0.0,
        "fusion_rank": inv(cand.get("rank")),
        "lex_overlap": len(inter) / len(union) if union else 0.0,
        "key_coverage": len(inter & kt) / len(kt) if kt else 0.0,
        "scope_overlap": len(inter & scope) / len(scope) if scope else 0.0,
    }


def normalize(vals):
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [1.0] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


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
        1 for q in ranked if any(
            labels[(q, c)] == "relevant" for c in ranked[q][:5])) / n
    return m


def build_pools(can, lab, topk=10):
    """Return {qid: [ (chunk_id, cand), ... ]} for the frozen top-k pool."""
    by_q = defaultdict(list)
    for (q, c), r in can.items():
        by_q[q].append((c, r))
    pools = {}
    for q, items in by_q.items():
        items.sort(key=lambda x: x[1].get("reranker_rank", 10**9))
        pools[q] = items[:topk]
    return pools


def reorder(pool, alpha, signal_name, signal_values):
    """Reorder a question's top-k pool with combined = (1-a)*norm_ce + a*norm_sig."""
    ranked = []
    for (cid, cand) in pool:
        ranked.append((cid, cand))
    if alpha == 0.0:
        # frozen order: already sorted by reranker_rank
        return [cid for cid, _ in pool]

    ce = normalize([c["reranker_score"] for _, c in pool])
    sig = normalize([signal_values[cid] for cid, _ in pool])
    combined = [
        (1.0 - alpha) * a + alpha * b for a, b in zip(ce, sig)
    ]
    order = sorted(range(len(pool)), key=lambda i: (-combined[i], i))
    return [pool[i][0] for i in order]


def main():
    datasets = [
        ("bench", BENCH_CANDS, BENCH_LABELS),
        ("hold", HOLDOUT_CANDS, HOLDOUT_LABELS),
    ]

    results = {}
    for setname, cands_path, labels_path in datasets:
        can, lab = load_pairs(cands_path, labels_path)
        pools = build_pools(can, lab, topk=10)
        qids = sorted(pools.keys())

        # frozen order baseline
        frozen = {q: [cid for cid, _ in pools[q]] for q in qids}
        results[f"{setname}_baseline"] = metrics(frozen, lab)

        # precompute per-question secondary signal values
        per_q_sig = {}
        for q, items in pools.items():
            qtext = items[0][1].get("question_text", "")
            per_q_sig[q] = {
                sig: {cid: secondary_features(c, qtext)[sig] for cid, c in items}
                for sig in SECONDARY_SIGNALS
            }

        for sig in SECONDARY_SIGNALS:
            for alpha in ALPHAS:
                key = f"{sig}_a{alpha}"
                orders = {}
                for q in qids:
                    orders[q] = reorder(pools[q], alpha, sig, per_q_sig[q][sig])
                results[f"{setname}_{key}"] = metrics(orders, lab)

                # save orders
                with open(f"{OUTDIR}/{setname}_{key}_order.jsonl", "w") as f:
                    for q in qids:
                        f.write(json.dumps(
                            {"question_id": q, "order": orders[q],
                             "signal": sig, "alpha": alpha}) + "\n")

    # build a compact summary
    summary = {"baseline": {
        "bench": results["bench_baseline"], "hold": results["hold_baseline"]}}
    for sig in SECONDARY_SIGNALS:
        for alpha in ALPHAS:
            summary[f"{sig}_a{alpha}"] = {
                "bench": results[f"bench_{sig}_a{alpha}"],
                "hold": results[f"hold_{sig}_a{alpha}"],
                "bench_delta": {
                    k: round(results[f"bench_{sig}_a{alpha}"][k]
                             - results["bench_baseline"][k], 4)
                    for k in results["bench_baseline"]},
                "hold_delta": {
                    k: round(results[f"hold_{sig}_a{alpha}"][k]
                             - results["hold_baseline"][k], 4)
                    for k in results["hold_baseline"]},
            }
    with open(f"{OUTDIR}/summary_ideaB.json", "w") as f:
        json.dump(summary, f, indent=1)

    # console report
    print(f"{'signal':<14} {'a':<5} | "
          f"{'bP@1':<7}{'bP@3':<7}{'bP@5':<7}{'bMRR':<8}{'hP@1':<7}{'hP@3':<7}{'hP@5':<7}{'hMRR':<8}")
    print(f"{'baseline':<14} {'-':<5} | "
          f"{results['bench_baseline']['P@1']:<7.3f}{results['bench_baseline']['P@3']:<7.3f}"
          f"{results['bench_baseline']['P@5']:<7.3f}{results['bench_baseline']['MRR']:<8.3f}"
          f"{results['hold_baseline']['P@1']:<7.3f}{results['hold_baseline']['P@3']:<7.3f}"
          f"{results['hold_baseline']['P@5']:<7.3f}{results['hold_baseline']['MRR']:<8.3f}")
    for sig in SECONDARY_SIGNALS:
        for alpha in ALPHAS:
            k = f"{sig}_a{alpha}"
            b = results[f"bench_{k}"]
            h = results[f"hold_{k}"]
            print(f"{sig:<14} {alpha:<5.1f} | "
                  f"{b['P@1']:<7.3f}{b['P@3']:<7.3f}{b['P@5']:<7.3f}{b['MRR']:<8.3f}"
                  f"{h['P@1']:<7.3f}{h['P@3']:<7.3f}{h['P@5']:<7.3f}{h['MRR']:<8.3f}")


if __name__ == "__main__":
    main()
