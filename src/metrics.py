"""Retrieval metrics: Precision@K computed from labeled results (Phase 12.6/12.7).

Pure computation on the already-labeled JSONL. No LLM, no retrieval, no Chroma.
"""


def validate_labeled(records):
    """Validate a labeled results set; raise ValueError with a clear message.

    Checks:
    - all 20 question IDs present, each with exactly ranks 1..10
    - no duplicate (question_id, rank) pairs
    - no null relevance labels (null is NOT treated as not_relevant)
    """
    seen = {}
    for record in records:
        qid = record.get("question_id")
        rank = record.get("rank")
        relevant = record.get("relevant")

        if qid is None or rank is None:
            raise ValueError(f"record missing question_id/rank: {record!r}")
        if relevant is None:
            raise ValueError(
                f"record {qid} rank {rank} has 'relevant': null — null is NOT "
                f"treated as not_relevant; label it first"
            )
        if relevant not in ("relevant", "not_relevant"):
            raise ValueError(
                f"record {qid} rank {rank} has invalid relevance: {relevant!r}"
            )

        key = (qid, rank)
        if key in seen:
            raise ValueError(f"duplicate (question_id, rank): {key}")
        seen[key] = True

    qids = {r["question_id"] for r in records}
    if len(qids) != 20:
        raise ValueError(
            f"expected 20 questions, found {len(qids)}: {sorted(qids)}"
        )

    for qid in qids:
        ranks = sorted(r["rank"] for r in records if r["question_id"] == qid)
        if ranks != list(range(1, 11)):
            raise ValueError(
                f"question {qid} must have ranks 1..10, found: {ranks}"
            )


def precision_at_k(records, k=3):
    """Per-question and mean Precision@K over a labeled results set.

    Precision@K(question) = (relevant among ranks 1..K) / K
    Mean P@K             = sum(per-question P@K) / number of questions

    Returns (per_question, mean):
      per_question: {qid: {"precision": float, "relevant": int, "n": int}}
    """
    by_question = {}
    for record in records:
        by_question.setdefault(record["question_id"], []).append(record)

    per_question = {}
    for qid, question_records in sorted(by_question.items()):
        ranked = sorted(question_records, key=lambda r: r["rank"])
        top_k = ranked[:k]
        relevant = sum(1 for r in top_k if r["relevant"] == "relevant")
        per_question[qid] = {
            "precision": relevant / k,
            "relevant": relevant,
            "n": len(top_k),
        }

    mean = sum(v["precision"] for v in per_question.values()) / len(per_question)
    return per_question, mean


def relevant_at_10(records):
    """Number of relevant chunks in the full Top-10, per question.

    For reporting only (not used by Precision@3/@5).
    """
    by_question = {}
    for record in records:
        by_question.setdefault(record["question_id"], []).append(record)

    out = {}
    for qid, question_records in sorted(by_question.items()):
        out[qid] = sum(1 for r in question_records if r["relevant"] == "relevant")
    return out


def topk_analysis(records):
    """Rank-distribution analysis of relevant chunks (Phase 12.8).

    For each question: first relevant rank, relevant counts in Top-3/5/10,
    has-relevant flags, P@3/P@5/P@10, and the relevance of the added ranks
    (4-5 vs top-3, 6-10 vs top-5).

    Returns (per_question, aggregates):
      per_question: {qid: {...}} sorted by question id
      aggregates:   dict of cross-question statistics
    """
    validate_labeled(records)

    by_question = {}
    for record in records:
        by_question.setdefault(record["question_id"], []).append(record)

    per_question = {}
    for qid, question_records in sorted(by_question.items()):
        ranked = sorted(question_records, key=lambda r: r["rank"])
        ranks = [r["relevant"] for r in ranked]

        def count_at(k):
            return sum(1 for lab in ranks[:k] if lab == "relevant")

        rel_positions = [
            i for i, lab in enumerate(ranks, start=1) if lab == "relevant"
        ]
        per_question[qid] = {
            "question_id": qid,
            "first_relevant_rank": rel_positions[0] if rel_positions else None,
            "relevant_count_at_3": count_at(3),
            "relevant_count_at_5": count_at(5),
            "relevant_count_at_10": count_at(10),
            "has_relevant_at_3": count_at(3) > 0,
            "has_relevant_at_5": count_at(5) > 0,
            "has_relevant_at_10": count_at(10) > 0,
            "precision_at_3": count_at(3) / 3,
            "precision_at_5": count_at(5) / 5,
            "precision_at_10": count_at(10) / 10,
            "ranks_4_5_relevance": ranks[3:5],
            "ranks_6_10_relevance": ranks[5:10],
        }

    qids = list(per_question)
    n = len(qids)
    rel3 = [q for q in qids if per_question[q]["has_relevant_at_3"]]
    rel5 = [q for q in qids if per_question[q]["has_relevant_at_5"]]
    rel10 = [q for q in qids if per_question[q]["has_relevant_at_10"]]

    distribution = {str(rank): 0 for rank in range(1, 11)}
    distribution["none"] = 0
    for q in qids:
        first = per_question[q]["first_relevant_rank"]
        if first is None:
            distribution["none"] += 1
        else:
            distribution[str(first)] += 1

    aggregates = {
        "questions_with_relevant_at_3": len(rel3),
        "questions_with_relevant_at_5": len(rel5),
        "questions_with_relevant_at_10": len(rel10),
        "pct_with_relevant_at_3": len(rel3) / n,
        "pct_with_relevant_at_5": len(rel5) / n,
        "pct_with_relevant_at_10": len(rel10) / n,
        "questions_with_no_relevant_at_10": distribution["none"],
        "first_relevant_rank_distribution": distribution,
    }
    return per_question, aggregates
