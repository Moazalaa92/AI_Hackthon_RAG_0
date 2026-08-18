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

from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker(model_name=CROSS_ENCODER_MODEL):
    """Return a lazily-loaded, shared CrossEncoder instance."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(model_name, max_length=512)
    return _reranker


def rerank_candidates(candidates, model_name=CROSS_ENCODER_MODEL):
    """Score every (question_text, chunk_text) pair with the cross-encoder.

    Returns NEW records (input records are not modified) with:
      reranker_score : raw cross-encoder logit (higher = more relevant)
      reranker_rank  : 1..n position after sorting the question's union by
                       cross-encoder score (the final ranking signal)
    """
    reranker = get_reranker(model_name)
    by_question = {}
    for record in candidates:
        by_question.setdefault(record["question_id"], []).append(record)

    out = []
    for qid, records in sorted(by_question.items()):
        question_text = records[0]["question_text"]
        pairs = [(question_text, r["chunk_text"]) for r in records]
        scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        scored = list(zip(records, scores))
        scored.sort(key=lambda item: -float(item[1]))
        for reranked_rank, (record, score) in enumerate(scored, start=1):
            new_record = dict(record)
            new_record["reranker_score"] = float(score)
            new_record["reranker_rank"] = reranked_rank
            out.append(new_record)
    return out