"""CLI: sweep final-stage ranking strategies over a LABELED candidate pool.

Every candidate in a labeled pool already carries a relevance label, so any
re-ordering of that pool is directly scorable. This makes ranking experiments
free: no retrieval, no LLM judge, no new labels. Only strategies that change
the CANDIDATE POOL (chunking, embedding, retrieval depth) need re-judging and
are out of scope for this script.

Strategies compared:
  rrf        - the historical RRF fusion order (the pool's own `rank`)
  dense      - dense L2 distance only (from provenance)
  bm25       - BM25 score only (from provenance)
  <model>    - a single cross-encoder
  ensemble   - mean of per-question z-standardized cross-encoder scores
  +w*bm25    - ensemble/model blended with the z-standardized BM25 score

Usage:
    python scripts/sweep_rankers.py \
        evaluation/results/hybrid_800_100_candidates_labeled.jsonl \
        evaluation/results/holdout_v1_hybrid_candidates_labeled.jsonl

Reported per pool: the achievable P@3/P@5 ceiling (limited by how many chunks
in the pool are labeled relevant), then P@3, P@5, Hit@3 and MRR per strategy.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR
from src.reranking import CROSS_ENCODER_MODEL, score_candidates, zscore

DEFAULT_MODELS = [
    CROSS_ENCODER_MODEL,
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3",
]
BM25_WEIGHTS = [0.1, 0.2, 0.3]


def read_records(path):
    """Load a labeled candidate-pool JSONL; exit on a missing/malformed file."""
    if not Path(path).exists():
        sys.exit(f"Error: labeled pool not found: {path}")
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
    for r in records:
        if r.get("relevant") not in ("relevant", "not_relevant"):
            sys.exit(
                f"Error: {path} is not fully labeled (question {r.get('question_id')} "
                f"rank {r.get('rank')} has relevant={r.get('relevant')!r}). "
                f"Run scripts/label_results.py first."
            )
    return records


def group(records):
    """{question_id: [records in pool order]} (pool order = the stored rank)."""
    by_question = {}
    for record in records:
        by_question.setdefault(record["question_id"], []).append(record)
    for recs in by_question.values():
        recs.sort(key=lambda r: r["rank"])
    return by_question


def metrics(ranked_by_question):
    """Mean P@3, P@5, Hit@3 and MRR over an ordering of each question's pool."""
    n = len(ranked_by_question)
    p3 = p5 = hit3 = mrr = 0.0
    for ranked in ranked_by_question.values():
        labels = [r["relevant"] == "relevant" for r in ranked]
        p3 += sum(labels[:3]) / 3
        p5 += sum(labels[:5]) / 5
        hit3 += any(labels[:3])
        for position, label in enumerate(labels, start=1):
            if label:
                mrr += 1 / position
                break
    return {"P@3": p3 / n, "P@5": p5 / n, "Hit@3": hit3 / n, "MRR": mrr / n}


def ceiling(by_question):
    """Best P@3/P@5 any ranker could reach given the labels in the pool."""
    n = len(by_question)
    counts = [
        sum(1 for r in recs if r["relevant"] == "relevant")
        for recs in by_question.values()
    ]
    return {
        "P@3": sum(min(c, 3) / 3 for c in counts) / n,
        "P@5": sum(min(c, 5) / 5 for c in counts) / n,
    }


def order_by(by_question, score_fn):
    """Re-order every question's pool by a per-question scoring function."""
    ordered = {}
    for qid, recs in by_question.items():
        scored = sorted(zip(recs, score_fn(qid, recs)), key=lambda item: -item[1])
        ordered[qid] = [r for r, _ in scored]
    return ordered


def strategies(by_question, model_scores, models):
    """Yield (label, ordering) for every ranking strategy under comparison."""
    yield "rrf (current pool order)", dict(by_question)
    yield "dense only", order_by(
        by_question,
        lambda q, recs: [-(r["dense_score"] or 1e9) for r in recs],
    )
    yield "bm25 only", order_by(
        by_question, lambda q, recs: [r["bm25_score"] or 0.0 for r in recs]
    )

    for model in models:
        yield model, order_by(by_question, lambda q, recs, m=model: model_scores[m][q])

    if len(models) > 1:
        def ensemble(qid, recs, chosen):
            stacked = [zscore(model_scores[m][qid]) for m in chosen]
            return [sum(col) / len(chosen) for col in zip(*stacked)]

        yield "ensemble (all models)", order_by(
            by_question, lambda q, recs: ensemble(q, recs, models)
        )
        best_pair = models[:1] + models[-1:]
        yield f"ensemble ({' + '.join(best_pair)})", order_by(
            by_question, lambda q, recs: ensemble(q, recs, best_pair)
        )
        for weight in BM25_WEIGHTS:
            yield f"ensemble(pair) + {weight}*bm25", order_by(
                by_question,
                lambda q, recs, w=weight: [
                    e + w * b
                    for e, b in zip(
                        ensemble(q, recs, best_pair),
                        zscore([r["bm25_score"] or 0.0 for r in recs]),
                    )
                ],
            )


def main():
    parser = argparse.ArgumentParser(
        description="Sweep final-stage ranking strategies over labeled candidate pools."
    )
    parser.add_argument("pools", nargs="+", help="Labeled candidate-pool JSONL files")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Cross-encoder models to compare (first + last are also ensembled)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(METRICS_DIR) / "ranker_sweep.json"),
        help="Where to write the sweep results JSON",
    )
    args = parser.parse_args()

    summary = {}
    for pool_path in args.pools:
        records = read_records(pool_path)
        by_question = group(records)
        model_scores = {m: score_candidates(records, m) for m in args.models}

        name = Path(pool_path).stem
        limit = ceiling(by_question)
        print(f"\n=== {name} ({len(by_question)} questions) ===")
        print(
            f"achievable ceiling given the labels: "
            f"P@3={limit['P@3']:.4f}  P@5={limit['P@5']:.4f}"
        )
        header = f"{'strategy':52s} {'P@3':>7s} {'P@5':>7s} {'Hit@3':>7s} {'MRR':>7s}"
        print(header)
        print("-" * len(header))

        results = {}
        for label, ordering in strategies(by_question, model_scores, args.models):
            values = metrics(ordering)
            results[label] = values
            print(
                f"{label:52s} {values['P@3']:7.4f} {values['P@5']:7.4f} "
                f"{values['Hit@3']:7.4f} {values['MRR']:7.4f}"
            )
        summary[name] = {"n_questions": len(by_question), "ceiling": limit,
                         "strategies": results}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nWrote sweep results to {out_path}")


if __name__ == "__main__":
    main()
