"""Phase 10 STEP 5 — aggregate LLM diagnostic judgments.

Reads step5_diagnosis.jsonl (deduped, last record wins) and produces:
  - diagnostic class counts (benchmark vs holdout)
  - existing relevance label vs LLM usefulness comparison
  - P@3 failure table with score margins
  - reranker score-margin analysis by failure type

Output: evaluation/experiments/step5_summary.json + console tables
"""

import json
import os

DIAG = "evaluation/experiments/step5_diagnosis.jsonl"


def load():
    best = {}
    for line in open(DIAG):
        r = json.loads(line)
        best[(r["question_id"], r["chunk_id"])] = r
    return best


def main():
    best = load()
    rows = list(best.values())
    print(f"records: {len(rows)}")

    sets = {"BENCH": [r for r in rows if r["question_id"].startswith("Q")],
            "HOLD": [r for r in rows if r["question_id"].startswith("H")]}
    from collections import Counter

    summary = {"records": len(rows)}
    for name, rs in sets.items():
        cls = Counter(r["judgment"]["diagnostic_class"] for r in rs)
        usf = Counter(r["judgment"]["answer_usefulness"] for r in rs)
        rel = Counter(r["existing_relevance"] for r in rs)
        summary[name] = {"n": len(rs), "class": dict(cls), "usefulness": dict(usf),
                         "existing_relevance": dict(rel)}
        print(f"\n=== {name} (n={len(rs)}) ===")
        print("class:", dict(cls))
        print("usefulness:", dict(usf))
        print("existing_relevance:", dict(rel))

    # Existing label vs usefulness confusion
    print("\n=== existing relevance vs LLM answer_usefulness (all) ===")
    cm = Counter()
    for r in rows:
        cm[(r["existing_relevance"], r["judgment"]["answer_usefulness"])] += 1
    for k in sorted(cm):
        print(f"  label={k[0]:<13} usefulness={k[1]:<6} count={cm[k]}")

    # 'relevant but not a direct answer' cases
    print("\n=== relevant-labeled chunks whose diagnostic_class is NOT DIRECT_ANSWER ===")
    n_relevant_not_direct = 0
    for r in rows:
        if r["existing_relevance"] == "relevant" and r["judgment"]["diagnostic_class"] != "DIRECT_ANSWER":
            n_relevant_not_direct += 1
            if r["in_top10"] and r["reranker_rank"] <= 5:
                print(f"  {r['question_id']} rank{r['reranker_rank']} {r['judgment']['diagnostic_class']:<22} "
                      f"useful={r['judgment']['answer_usefulness']} score={r['reranker_score']:.2f}")
    print(f"total relevant-but-not-direct-answer: {n_relevant_not_direct}")

    summary["relevant_not_direct_answer"] = n_relevant_not_direct

    with open("evaluation/experiments/step5_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nwrote evaluation/experiments/step5_summary.json")


if __name__ == "__main__":
    main()