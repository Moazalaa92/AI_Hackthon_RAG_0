"""Cross-encoder reranking over the hybrid candidate pool (final ranking stage).

A cross-encoder evaluates the full (question, chunk) pair jointly, unlike the
bi-encoder dense retrieval which encodes question and chunk independently and
scores them with a cosine/L2 similarity of the two embeddings. The hypothesis
under test: a cross-encoder can examine the relationship between the exact
question and each candidate chunk more precisely than the initial embedding
retrieval, and thus move genuinely relevant chunks toward the top.

The cross-encoder is the SOLE final ranking signal: the candidate union
(dense Top-20 + BM25 Top-20) is scored and sorted by the raw cross-encoder
logit (higher = stronger predicted relevance). RRF is deliberately NOT used in
the final ranking: RRF was a rank-based fusion for combining retrievers
without a learned scorer, and it structurally capped single-retriever chunks
at ~1/(k+1) (the Q09 failure: BM25-only relevant chunks fused to ranks
13/27). With a supervised relevance scorer available, the direct
(question, chunk) score replaces RRF as the ranking mechanism.
"""

from threading import RLock

from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_rerankers = {}
_reranker_lock = RLock()


def get_reranker(model_name=CROSS_ENCODER_MODEL):
    """Return a lazily-loaded, shared CrossEncoder instance per model name."""
    with _reranker_lock:
        if model_name not in _rerankers:
            _rerankers[model_name] = CrossEncoder(model_name, max_length=512)
        return _rerankers[model_name]


def zscore(values):
    """Standardize scores within one question's candidate pool.

    Cross-encoders have incomparable score scales (raw logits), so scores are
    only combined after per-question standardization.
    """
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 or 1.0
    return [(v - mean) / sd for v in values]


def score_candidates(candidates, model_name=CROSS_ENCODER_MODEL):
    """Return {question_id: [score, ...]} aligned with the input record order."""
    reranker = get_reranker(model_name)
    by_question = {}
    for record in candidates:
        by_question.setdefault(record["question_id"], []).append(record)

    scores = {}
    for qid, records in sorted(by_question.items()):
        question_text = records[0]["question_text"]
        pairs = [(question_text, r["chunk_text"]) for r in records]
        raw = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        scores[qid] = [float(s) for s in raw]
    return scores


def rerank_candidates(candidates, model_name=CROSS_ENCODER_MODEL,
                      model_names=None, bm25_weight=0.0):
    """Score every (question_text, chunk_text) pair and rank the union.

    `model_names` runs an ensemble: each model's scores are z-standardized
    within a question and averaged. `bm25_weight` adds the z-standardized BM25
    score from the candidate record's provenance, which slightly favours
    lexically exact evidence. Both default to the single-model, no-blend
    behaviour of the frozen architecture.

    Returns NEW records (input records are not modified) with:
      reranker_score : final ranking score (higher = more relevant); the raw
                       cross-encoder logit for a single model with no blend
      reranker_rank  : 1..n position after sorting the question's union by
                       reranker_score (the final ranking signal)
    """
    models = model_names or [model_name]
    per_model = {m: score_candidates(candidates, m) for m in models}

    by_question = {}
    for record in candidates:
        by_question.setdefault(record["question_id"], []).append(record)

    single_raw = len(models) == 1 and not bm25_weight
    out = []
    for qid, records in sorted(by_question.items()):
        if single_raw:
            combined = per_model[models[0]][qid]
        else:
            stacked = [zscore(per_model[m][qid]) for m in models]
            combined = [sum(col) / len(models) for col in zip(*stacked)]
            if bm25_weight:
                bm25 = zscore([r.get("bm25_score") or 0.0 for r in records])
                combined = [c + bm25_weight * b for c, b in zip(combined, bm25)]

        scored = sorted(zip(records, combined), key=lambda item: -item[1])
        for reranked_rank, (record, score) in enumerate(scored, start=1):
            new_record = dict(record)
            new_record["reranker_score"] = float(score)
            new_record["reranker_rank"] = reranked_rank
            out.append(new_record)
    return out