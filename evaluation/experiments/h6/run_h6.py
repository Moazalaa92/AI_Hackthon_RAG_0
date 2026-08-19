import json
import re
import os
from collections import defaultdict


F_SIGNALS = {
    "F1_toc": re.compile(r"\.{6,}"),
    "F2_research": re.compile(r"recommendation for research", re.I),
    "F3_ta": re.compile(r"nice technology appraisal guidance", re.I),
    "F4_affect": re.compile(r"how the recommendations might affect practice", re.I),
    "F5_evidx": re.compile(r"^\s*\u2022?\s*evidence review\s+[A-Z0-9]+:", re.I | re.M),
    "F6_why": re.compile(r"why the committee made these recommendations", re.I),
}

NUM_REC = re.compile(r"^\s*\d+\.\d+(\.\d+)?\s", re.M)
ACTION = re.compile(r"\b(offer|consider|refer|monitor|recommend|treat|do not|use|ensure|offer a choice of|should be)\b", re.I)
MEDICINE = re.compile(
    r"\b(lamotrigine|levetiracetam|valproate|ethosuximide|vigabatrin|prednisolone|"
    r"clobazam|fenfluramine|carbamazepine|topiramate|zonisamide|perampanel|rufinamide|"
    r"diazepam|midazolam|dexamethasone|nitrazepam|steroid)\b", re.I)

# Scope qualifiers (Q11-style). Only the specific, safe phrases.
SCOPE_QUALIFIERS = re.compile(
    r"with other seizure types|or at risk of these|second-line|add-on treatment|"
    r"if first-line treatment is unsuccessful|if first-line treatments?", re.I)

# Weights applied by each variant. Only signals with 0% relevant-rate (F1/F2/F3)
# are applied unconditionally. F4/F5/F6 and scope are tested but shown to cause
# regressions (relevant chunks share the same text patterns), so they are NOT in
# the recommended H6-A.
PENALTIES = {
    "F1_toc": 3.0,
    "F2_research": 3.0,
    "F3_ta": 2.0,
    "F4_affect": 1.5,
    "F5_evidx": 1.5,
    "F6_why": 1.0,
}


def signals(chunk_text):
    return [name for name, pat in F_SIGNALS.items() if pat.search(chunk_text)]


def is_protected(chunk_text):
    return bool(NUM_REC.search(chunk_text)) or (bool(ACTION.search(chunk_text)) and bool(MEDICINE.search(chunk_text)))


def has_scope_qualifier(chunk_text):
    return bool(SCOPE_QUALIFIERS.search(chunk_text))


def core_penalty(chunk, rank, q_text):
    """F1/F2/F3 only: the safe, high-precision index/rationale suppression."""
    sigs = set(signals(chunk["chunk_text"]))
    p = 0.0
    if "F1_toc" in sigs:
        p -= PENALTIES["F1_toc"]
    if "F2_research" in sigs:
        p -= PENALTIES["F2_research"]
    if "F3_ta" in sigs:
        p -= PENALTIES["F3_ta"]
    return p


def variant_a(chunk, rank, q_text, orig_score, ctx=None):
    return orig_score + core_penalty(chunk, rank, q_text)


def variant_b(chunk, rank, q_text, orig_score, ctx=None):
    """core + limited diversity: demote additional F-flagged unprotected chunks
    that share a page with another F-flagged chunk. Shown to add no benchmark
    value and slightly hurt holdout P@5."""
    p = core_penalty(chunk, rank, q_text)
    if ctx is not None:
        sigs = set(signals(chunk["chunk_text"]))
        prot = is_protected(chunk["chunk_text"])
        if sigs and not prot:
            if ctx.get((chunk["page_label"], True), 0) > 1:
                p -= 1.0
    return orig_score + p


def variant_c(chunk, rank, q_text, orig_score, ctx=None):
    """core + limited diversity + scope-mismatch penalty. The scope penalty is
    NOT recommended: it regresses Q16/H08/H23 because relevant chunks share the
    same scope phrases (e.g. Q11's relevant chunk contains 'add-on treatment')."""
    p = variant_b(chunk, rank, q_text, orig_score, ctx) - orig_score
    if has_scope_qualifier(chunk["chunk_text"]) and not has_scope_qualifier(q_text):
        p -= 0.75
    return orig_score + p


VARIANTS = {
    "baseline": None,
    "H6-A": variant_a,
    "H6-B": variant_b,
    "H6-C": variant_c,
}


def load(name):
    return [json.loads(line) for line in open(name)]


def rerank(cands, variant):
    out = {}
    for q in sorted(set(c["question_id"] for c in cands)):
        rows = [c for c in cands if c["question_id"] == q]
        if variant is None:
            rows.sort(key=lambda c: c["reranker_score"], reverse=True)
        else:
            ctx = defaultdict(int)
            for c in rows:
                sigs = set(signals(c["chunk_text"]))
                prot = is_protected(c["chunk_text"])
                if sigs and not prot:
                    ctx[(c["page_label"], True)] += 1
            rows.sort(key=lambda c: c["reranker_rank"])
            scored = []
            for r, c in enumerate(rows, 1):
                adj = variant(c, r, c["question_text"], c["reranker_score"], dict(ctx))
                scored.append((adj, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            rows = [c for _, c in scored]
        out[q] = rows
    return out


def metrics(ranked, labels, ks=(1, 3, 5)):
    m = {}
    n = len(ranked)
    for k in ks:
        m[f"P@{k}"] = sum(
            sum(1 for c in ranked[q][:k] if labels[(q, c["chunk_id"])] == "relevant") / k
            for q in ranked) / n
    m["MRR"] = sum(
        1 / next((i for i, c in enumerate(ranked[q], 1)
                  if labels[(q, c["chunk_id"])] == "relevant"), 1e9)
        for q in ranked) / n
    m["Hit@3"] = sum(
        1 for q in ranked if any(labels[(q, c["chunk_id"])] == "relevant" for c in ranked[q][:3])) / n
    m["Hit@5"] = sum(
        1 for q in ranked if any(labels[(q, c["chunk_id"])] == "relevant" for c in ranked[q][:5])) / n
    return m


def main():
    bench = load("evaluation/results/reranked_hybrid_800_100_candidates.jsonl")
    bench_labels = {}
    for line in open("evaluation/results/hybrid_800_100_candidates_labeled.jsonl"):
        r = json.loads(line)
        bench_labels[(r["question_id"], r["chunk_id"])] = r["relevant"]

    outdir = "evaluation/experiments/h6/results"
    os.makedirs(outdir, exist_ok=True)

    summary = {}
    for name, fn in VARIANTS.items():
        ranked = rerank(bench, fn)
        summary[name] = metrics(ranked, bench_labels)
        with open(f"{outdir}/{name}_per_question.jsonl", "w") as f:
            for q, rows in ranked.items():
                f.write(json.dumps({"question_id": q, "chunks": [
                    {"chunk_id": c["chunk_id"], "page_label": c["page_label"],
                     "reranker_score": round(c["reranker_score"], 3), "relevant": bench_labels[(q, c["chunk_id"])]}
                    for c in rows[:10]]}) + "\n")

    print("BENCHMARK")
    print(json.dumps(summary, indent=2))
    with open(f"{outdir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    hol = load("evaluation/results/holdout_v1_reranked_hybrid_candidates.jsonl")
    hol_labels = {}
    for line in open("evaluation/results/holdout_v1_hybrid_candidates_labeled.jsonl"):
        r = json.loads(line)
        hol_labels[(r["question_id"], r["chunk_id"])] = r["relevant"]
    hol_summary = {}
    for name, fn in VARIANTS.items():
        ranked = rerank(hol, fn)
        hol_summary[name] = metrics(ranked, hol_labels)
    print("HOLDOUT")
    print(json.dumps(hol_summary, indent=2))
    with open(f"{outdir}/holdout_summary.json", "w") as f:
        json.dump(hol_summary, f, indent=2)


if __name__ == "__main__":
    main()