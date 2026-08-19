"""Phase 10 STEP 6 — second-stage answerability reorder experiment (OFFLINE).

Determines whether an answerability/usefulness signal can improve the ordering
of the frozen reranked Top-10 results, evaluated ONLY against the frozen binary
relevance labels (P@1/P@3/P@5/MRR/Hit@5).

Scope of reordering: the frozen Top-10 per question (never removes chunks, never
pulls in chunks from beyond top-10, never touches the candidate pool or the
frozen pipeline). Non-affected questions (no LLM diagnostic signal) keep the
frozen order.

Variants:
  baseline            frozen order (must reproduce canonical baselines)
  oracle_usefulness   LLM answer_usefulness (high/medium/low) used directly
                      as the answerability score  -> ORACLE / UPPER-BOUND
  oracle_class        LLM diagnostic_class mapped to a score -> ORACLE
  lqo                 leave-question-out regression classifier trained on the
                      diagnostic dataset (all chunks from OTHER questions),
                      predicts usefulness for the held-out question's top-10
                      -> NON-ORACLE
  crossset            train on benchmark affected questions, apply to holdout
                      affected questions (and vice versa) -> NON-ORACLE
  suppress_rationale  hard demotion of RATIONALE/INDEX_TOC within top-10

Each non-baseline variant is tested over an alpha sweep of the combined score:
    combined = (1 - alpha) * norm_ce + alpha * norm_answerability
where norm_ce is per-question min-max of reranker_score within the top-10 pool.

All outputs under evaluation/experiments/step6/results/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from sklearn.linear_model import LogisticRegression

OUTDIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTDIR, exist_ok=True)

BENCH_CANDS = "evaluation/results/reranked_hybrid_800_100_candidates.jsonl"
BENCH_LABELS = "evaluation/results/hybrid_800_100_candidates_labeled.jsonl"
HOLDOUT_CANDS = "evaluation/results/holdout_v1_reranked_hybrid_candidates.jsonl"
HOLDOUT_LABELS = "evaluation/results/holdout_v1_hybrid_candidates_labeled.jsonl"
DIAG = "evaluation/experiments/step5_diagnosis.jsonl"

USEFULNESS_MAP = {"high": 1.0, "medium": 0.5, "low": 0.0}
CLASS_MAP = {
    "DIRECT_ANSWER": 1.0,
    "SUPPORTING_DETAIL": 0.7,
    "RELEVANT_NOT_ANSWERING": 0.4,
    "RATIONALE": 0.1,
    "INDEX_TOC": 0.0,
    "WRONG_SUBTOPIC": 0.2,
    "SCOPE_MISMATCH": 0.2,
    "OTHER": 0.3,
}
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]


# ---------------------------------------------------------------- data loaders
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


def affected_questions(can, lab):
    """Questions where a relevant chunk sits below the top-3 (P@3 loss)."""
    affected = []
    for q in sorted(set(k[0] for k in can)):
        rows = [k for k in can if k[0] == q]
        rows.sort(key=lambda k: can[k]["reranker_rank"])
        top3 = rows[:3]
        rel3 = sum(1 for k in top3 if lab[k] == "relevant")
        missed = [k for k in rows[3:] if lab[k] == "relevant"]
        if rel3 < 3 and missed:
            affected.append(q)
    return affected


def frozen_top10(can, q):
    rows = [k for k in can if k[0] == q]
    rows.sort(key=lambda k: can[k]["reranker_rank"])
    return rows[:10]


# ---------------------------------------------------------------- metric utils
def metrics(ranked, labels):
    """ranked: {q: [chunk_id,...]}; returns P@1/3/5 MRR Hit@5."""
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


# ---------------------------------------------------------------- reordering
def norm_minmax(vals):
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def build_order(can, q, score_fn, alpha, suppress=None):
    """score_fn(chunk_record) -> answerability score in [0,1]; returns list of
    chunk_ids in new order (length 10, a permutation of the frozen top-10)."""
    top10 = frozen_top10(can, q)
    ce = [can[k]["reranker_score"] for k in top10]
    nce = norm_minmax(ce)
    ans = [score_fn(can[k]) for k in top10]
    nan = norm_minmax(ans)
    combined = [(1 - alpha) * c + alpha * a for c, a in zip(nce, nan)]
    order = sorted(range(10), key=lambda i: combined[i], reverse=True)
    if suppress:
        # hard demotion: push suppressed chunks to the bottom of the top-10
        suppressed = [i for i in order if suppress(can[top10[i]])]
        kept = [i for i in order if i not in suppressed]
        order = kept + suppressed
    return [top10[i][1] for i in order]


def run_variant(can, lab, questions, score_fn, alpha, suppress=None):
    ranked = {}
    for q in sorted(set(k[0] for k in can)):
        if q in questions:
            ranked[q] = build_order(can, q, score_fn, alpha, suppress)
        else:
            ranked[q] = [c[1] for c in frozen_top10(can, q)]
    return ranked


# ---------------------------------------------------------------- LLM-derived signals
def make_oracle_usefulness(diag):
    def fn(rec):
        j = diag.get((rec["question_id"], rec["chunk_id"]))
        if j is None:
            return 0.0
        return USEFULNESS_MAP[j["judgment"]["answer_usefulness"]]
    return fn


def make_oracle_class(diag):
    def fn(rec):
        j = diag.get((rec["question_id"], rec["chunk_id"]))
        if j is None:
            return 0.0
        return CLASS_MAP[j["judgment"]["diagnostic_class"]]
    return fn


# ---------------------------------------------------------------- non-oracle classifiers
_EMB = None
_EMB_CACHE = {}


def _embedder():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB


def _featurize(question_text, chunk_text):
    key = (question_text, chunk_text)
    if key not in _EMB_CACHE:
        emb = _embedder()
        q = emb.encode(question_text, normalize_embeddings=True)
        c = emb.encode(chunk_text, normalize_embeddings=True)
        _EMB_CACHE[key] = np.concatenate([q, c])
    return _EMB_CACHE[key]


def _train_lr(train_rows, seed=0):
    """Train a logistic regression predicting usefulness (high/med/low target
    on a 0..1 scale) from (question, chunk) embeddings."""
    X = np.stack([_featurize(r["question_text"], r["chunk_text"]) for r in train_rows])
    y = np.array([USEFULNESS_MAP[r["judgment"]["answer_usefulness"]] for r in train_rows])
    clf = LogisticRegression(C=0.5, max_iter=2000, random_state=seed)
    # binarize into 3 classes via rounding to nearest of {0.0,0.5,1.0}
    yc = np.round(2 * y).astype(int)
    clf.fit(X, yc)
    return clf


def _predict_score(clf, question_text, chunk_text):
    x = _featurize(question_text, chunk_text).reshape(1, -1)
    proba = clf.predict_proba(x)[0]
    # expected usefulness given predicted class probabilities
    return float(np.dot(proba, [0.0, 0.5, 1.0]))


def make_lqo_score(rows_by_question, target_q):
    """Train on all questions except target_q; returns a score_fn."""
    train_rows = [r for q, rs in rows_by_question.items() if q != target_q for r in rs]
    clf = _train_lr(train_rows)

    def fn(rec):
        return _predict_score(clf, rec["question_text"], rec["chunk_text"])
    return fn


def make_crossset_score(train_rows):
    clf = _train_lr(train_rows)

    def fn(rec):
        return _predict_score(clf, rec["question_text"], rec["chunk_text"])
    return fn


# ---------------------------------------------------------------- suppression
def make_suppress_rationale(diag):
    def fn(rec):
        j = diag.get((rec["question_id"], rec["chunk_id"]))
        if j is None:
            return False
        return j["judgment"]["diagnostic_class"] in ("RATIONALE", "INDEX_TOC")
    return fn


# ---------------------------------------------------------------- analysis
def per_question_delta(before, after, labels, questions):
    rows = []
    for q in sorted(questions):
        def p3(r):
            return sum(1 for c in r[:3] if labels[(q, c)] == "relevant") / 3
        rows.append({
            "question_id": q,
            "P@3_before": round(p3(before[q]), 4),
            "P@3_after": round(p3(after[q]), 4),
            "P@1_before": int(labels[(q, before[q][0])] == "relevant"),
            "P@1_after": int(labels[(q, after[q][0])] == "relevant"),
            "P@5_before": round(sum(1 for c in before[q][:5] if labels[(q, c)] == "relevant") / 5, 4),
            "P@5_after": round(sum(1 for c in after[q][:5] if labels[(q, c)] == "relevant") / 5, 4),
        })
    return rows


def distractor_fixed(rows, label):
    n_improved = sum(1 for r in rows if r["P@3_after"] > r["P@3_before"])
    n_regressed = sum(1 for r in rows if r["P@3_after"] < r["P@3_before"])
    n_unchanged = len(rows) - n_improved - n_regressed
    return {"improved": n_improved, "unchanged": n_unchanged, "regressed": n_regressed,
            "label": label}


def main():
    bench_can, bench_lab = load_pairs(BENCH_CANDS, BENCH_LABELS)
    hold_can, hold_lab = load_pairs(HOLDOUT_CANDS, HOLDOUT_LABELS)
    diag = load_diagnosis()

    bench_affected = affected_questions(bench_can, bench_lab)
    hold_affected = affected_questions(hold_can, hold_lab)
    print("affected bench:", sorted(bench_affected))
    print("affected hold: ", sorted(hold_affected))

    # rows_by_question for classifier training: ALL diagnosed records
    rows_by_q = {}
    for (q, c), j in diag.items():
        rows_by_q.setdefault(q, []).append({
            "question_id": q, "question_text": j["question_text"],
            "chunk_id": c, "chunk_text": j["chunk_text"],
            "judgment": j["judgment"],
        })

    # ---------------------------------------------------------- baseline
    base_b = run_variant(bench_can, bench_lab, set(bench_affected), lambda r: 0.0, 0.0)
    base_h = run_variant(hold_can, hold_lab, set(hold_affected), lambda r: 0.0, 0.0)
    mb = metrics(base_b, bench_lab)
    mh = metrics(base_h, hold_lab)
    print("BENCH baseline:", {k: round(v, 4) for k, v in mb.items()})
    print("HOLD  baseline:", {k: round(v, 4) for k, v in mh.items()})

    results = {"baseline": {"bench": mb, "hold": mh},
               "bench_affected": sorted(bench_affected),
               "hold_affected": sorted(hold_affected)}

    # ---------------------------------------------------------- oracle variants
    oracle_ufn = make_oracle_usefulness(diag)
    oracle_cfn = make_oracle_class(diag)

    for name, score_fn in [("oracle_usefulness", oracle_ufn),
                           ("oracle_class", oracle_cfn)]:
        for alpha in ALPHAS:
            rb = run_variant(bench_can, bench_lab, set(bench_affected), score_fn, alpha)
            rh = run_variant(hold_can, hold_lab, set(hold_affected), score_fn, alpha)
            results[f"{name}_a{alpha}"] = {
                "bench": metrics(rb, bench_lab),
                "hold": metrics(rh, hold_lab),
                "bench_delta": {k: round(metrics(rb, bench_lab)[k] - mb[k], 4) for k in mb},
                "hold_delta": {k: round(metrics(rh, hold_lab)[k] - mh[k], 4) for k in mh},
            }
            # store reordered top-10 for per-question analysis
            with open(f"{OUTDIR}/{name}_a{alpha}_bench_order.jsonl", "w") as f:
                for q in sorted(set(k[0] for k in bench_can)):
                    f.write(json.dumps({"question_id": q, "order": rb[q]}) + "\n")
            with open(f"{OUTDIR}/{name}_a{alpha}_hold_order.jsonl", "w") as f:
                for q in sorted(set(k[0] for k in hold_can)):
                    f.write(json.dumps({"question_id": q, "order": rh[q]}) + "\n")

    # ---------------------------------------------------------- LQO (non-oracle)
    for alpha in ALPHAS:
        rb, rh = {}, {}
        for q in bench_affected:
            fn = make_lqo_score(rows_by_q, q)
            rb[q] = build_order(bench_can, q, fn, alpha)
        for q in hold_affected:
            fn = make_lqo_score(rows_by_q, q)
            rh[q] = build_order(hold_can, q, fn, alpha)
        # fill non-affected with frozen order
        for q in sorted(set(k[0] for k in bench_can)):
            rb.setdefault(q, [c[1] for c in frozen_top10(bench_can, q)])
        for q in sorted(set(k[0] for k in hold_can)):
            rh.setdefault(q, [c[1] for c in frozen_top10(hold_can, q)])
        results[f"lqo_a{alpha}"] = {
            "bench": metrics(rb, bench_lab),
            "hold": metrics(rh, hold_lab),
            "bench_delta": {k: round(metrics(rb, bench_lab)[k] - mb[k], 4) for k in mb},
            "hold_delta": {k: round(metrics(rh, hold_lab)[k] - mh[k], 4) for k in mh},
        }
        with open(f"{OUTDIR}/lqo_a{alpha}_bench_order.jsonl", "w") as f:
            for q in sorted(set(k[0] for k in bench_can)):
                f.write(json.dumps({"question_id": q, "order": rb[q]}) + "\n")
        with open(f"{OUTDIR}/lqo_a{alpha}_hold_order.jsonl", "w") as f:
            for q in sorted(set(k[0] for k in hold_can)):
                f.write(json.dumps({"question_id": q, "order": rh[q]}) + "\n")

    # ---------------------------------------------------------- cross-set (non-oracle)
    bench_rows = [r for q in bench_affected for r in rows_by_q[q]]
    hold_rows = [r for q in hold_affected for r in rows_by_q[q]]
    for alpha in ALPHAS:
        # train on bench, apply to hold
        fn_h = make_crossset_score(bench_rows)
        rh = {}
        for q in hold_affected:
            rh[q] = build_order(hold_can, q, fn_h, alpha)
        for q in sorted(set(k[0] for k in hold_can)):
            rh.setdefault(q, [c[1] for c in frozen_top10(hold_can, q)])
        # train on hold, apply to bench
        fn_b = make_crossset_score(hold_rows)
        rb = {}
        for q in bench_affected:
            rb[q] = build_order(bench_can, q, fn_b, alpha)
        for q in sorted(set(k[0] for k in bench_can)):
            rb.setdefault(q, [c[1] for c in frozen_top10(bench_can, q)])
        results[f"crossset_a{alpha}"] = {
            "bench": metrics(rb, bench_lab),
            "hold": metrics(rh, hold_lab),
            "bench_delta": {k: round(metrics(rb, bench_lab)[k] - mb[k], 4) for k in mb},
            "hold_delta": {k: round(metrics(rh, hold_lab)[k] - mh[k], 4) for k in mh},
        }
        with open(f"{OUTDIR}/crossset_a{alpha}_bench_order.jsonl", "w") as f:
            for q in sorted(set(k[0] for k in bench_can)):
                f.write(json.dumps({"question_id": q, "order": rb[q]}) + "\n")
        with open(f"{OUTDIR}/crossset_a{alpha}_hold_order.jsonl", "w") as f:
            for q in sorted(set(k[0] for k in hold_can)):
                f.write(json.dumps({"question_id": q, "order": rh[q]}) + "\n")

    # ---------------------------------------------------------- suppression
    sup = make_suppress_rationale(diag)
    rb = run_variant(bench_can, bench_lab, set(bench_affected), lambda r: 0.0, 0.0, suppress=sup)
    rh = run_variant(hold_can, hold_lab, set(hold_affected), lambda r: 0.0, 0.0, suppress=sup)
    results["suppress_rationale"] = {
        "bench": metrics(rb, bench_lab),
        "hold": metrics(rh, hold_lab),
        "bench_delta": {k: round(metrics(rb, bench_lab)[k] - mb[k], 4) for k in mb},
        "hold_delta": {k: round(metrics(rh, hold_lab)[k] - mh[k], 4) for k in mh},
    }
    with open(f"{OUTDIR}/suppress_rationale_bench_order.jsonl", "w") as f:
        for q in sorted(set(k[0] for k in bench_can)):
            f.write(json.dumps({"question_id": q, "order": rb[q]}) + "\n")
    with open(f"{OUTDIR}/suppress_rationale_hold_order.jsonl", "w") as f:
        for q in sorted(set(k[0] for k in hold_can)):
            f.write(json.dumps({"question_id": q, "order": rh[q]}) + "\n")

    # ---------------------------------------------------------- per-question analysis for selected variants
    pq = {}
    for name, score_fn, alpha, suppress in [
        ("oracle_usefulness_a0.3", oracle_ufn, 0.3, None),
        ("oracle_usefulness_a0.5", oracle_ufn, 0.5, None),
        ("lqo_a0.3", None, 0.3, None),
        ("crossset_a0.3", None, 0.3, None),
        ("suppress_rationale", None, 0.0, sup),
    ]:
        rb = {}
        for q in bench_affected:
            if score_fn is not None:
                fn = score_fn
            elif name.startswith("lqo"):
                fn = make_lqo_score(rows_by_q, q)
            elif name.startswith("crossset"):
                fn = make_crossset_score(hold_rows)
            else:
                fn = lambda r: 0.0
            rb[q] = build_order(bench_can, q, fn, alpha, suppress=suppress)
        rh = {}
        for q in hold_affected:
            if score_fn is not None:
                fn = score_fn
            elif name.startswith("lqo"):
                fn = make_lqo_score(rows_by_q, q)
            elif name.startswith("crossset"):
                fn = make_crossset_score(bench_rows)
            else:
                fn = lambda r: 0.0
            rh[q] = build_order(hold_can, q, fn, alpha, suppress=suppress)
        for q in sorted(set(k[0] for k in bench_can)):
            rb.setdefault(q, [c[1] for c in frozen_top10(bench_can, q)])
        for q in sorted(set(k[0] for k in hold_can)):
            rh.setdefault(q, [c[1] for c in frozen_top10(hold_can, q)])
        pq[name] = {
            "bench": per_question_delta(rb, rb, bench_lab, bench_affected),
            "hold": per_question_delta(rh, rh, hold_lab, hold_affected),
        }
        # careful: per_question_delta needs before=baseline order
    # recompute per-question with baseline as 'before'
    base_orders_b = {q: base_b[q] for q in bench_affected}
    base_orders_h = {q: base_h[q] for q in hold_affected}

    with open(f"{OUTDIR}/summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # print summary table
    print()
    print(f"{'variant':<28}{'P@1':>7}{'P@3':>7}{'P@5':>7}{'MRR':>7}{'Hit@5':>7}  |  "
          f"{'P@1Δ':>6}{'P@3Δ':>6}{'P@5Δ':>6}")
    for name, res in results.items():
        if name in ("baseline", "bench_affected", "hold_affected"):
            continue
        b = res["bench"]
        bd = res.get("bench_delta", {})
        print(f"{name:<28}{b['P@1']:7.3f}{b['P@3']:7.3f}{b['P@5']:7.3f}{b['MRR']:7.3f}{b['Hit@5']:7.3f}  |  "
              f"{bd.get('P@1', 0):+6.3f}{bd.get('P@3', 0):+6.3f}{bd.get('P@5', 0):+6.3f}")
    print()
    for name, res in results.items():
        if name in ("baseline", "bench_affected", "hold_affected"):
            continue
        h = res["hold"]
        hd = res.get("hold_delta", {})
        print(f"{name:<28}{h['P@1']:7.3f}{h['P@3']:7.3f}{h['P@5']:7.3f}{h['MRR']:7.3f}{h['Hit@5']:7.3f}  |  "
              f"{hd.get('P@1', 0):+6.3f}{hd.get('P@3', 0):+6.3f}{hd.get('P@5', 0):+6.3f}  (holdout)")

    print("\nwrote", f"{OUTDIR}/summary.json")


if __name__ == "__main__":
    main()