"""End-to-end RAG pipeline (Phase 10): retrieval -> generation -> citations.

Composes the frozen, individually-testable components into one entry point:

    query(question)
      -> retrieve_top_k()      (frozen hybrid retrieval + cross-encoder)
      -> generate_answer()     (frozen Generation v2)
      -> build_citations()     (claim-level citation layer, Phase 9)
      -> validate_citations()  (deterministic traceability checks)
      -> Answer

This module only ORCHESTRATES. It never re-implements retrieval, generation,
or citation logic; it reuses src.hybrid_retrieval, src.reranking,
src.generation, src.sources, and src.sources_judge unchanged.

Fail-closed policy on citation selection:
    If the citation selector raises SourceJudgeError (e.g. the LLM API is
    unavailable), the pipeline does NOT fall back to broader citation
    assignment (selector=None / all-candidates). It preserves the generated
    answer, sets citations_valid = False, populates validation_errors, and
    returns an Answer with no claims/citations rather than invent or widen
    them.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.embeddings import DEFAULT_EMBEDDING_MODEL
from src.generation import generate_answer
from src.hybrid_retrieval import BM25_CANDIDATES, DENSE_CANDIDATES, run_hybrid
from src.reranking import rerank_candidates
from src.sources import build_citations, validate_citations
from src.sources_judge import SourceJudgeError, select_supporting_chunks

DEFAULT_STORE = "data/chroma_db/experiments/large_800_100"
DEFAULT_FINAL_K = 10
QUESTION_ID = "ASK"

Selector = Callable[[str, str, list], list]


@dataclass
class Answer:
    """Result of one end-to-end query."""

    question: str
    answer: str
    claims: List[dict] = field(default_factory=list)
    retrieved_chunks: List[dict] = field(default_factory=list)
    citations_valid: Optional[bool] = None
    validation_errors: List[str] = field(default_factory=list)


def retrieve_top_k(
    question: str,
    persist_dir: str = DEFAULT_STORE,
    final_k: int = DEFAULT_FINAL_K,
) -> List[dict]:
    """Run the canonical frozen retrieval for a single question.

    Mirrors scripts/ask.py exactly: dense Top-20 + BM25 Top-20 -> union ->
    cross-encoder rerank -> keep the final Top-K chunk records.

    Args:
        question: the question to retrieve for.
        persist_dir: Chroma persist directory (default: canonical store).
        final_k: number of chunks to keep for context.

    Returns:
        List of chunk records ranked by the cross-encoder.
    """
    dataset = {"questions": [{"id": QUESTION_ID, "question": question}]}
    _, candidates = run_hybrid(
        dataset,
        persist_dir=persist_dir,
        model_name=DEFAULT_EMBEDDING_MODEL,
        dense_k=DENSE_CANDIDATES,
        bm25_k=BM25_CANDIDATES,
        final_k=final_k,
    )
    reranked = rerank_candidates(candidates)
    return [r for r in reranked if r["reranker_rank"] <= final_k]


def query(
    question: str,
    *,
    persist_dir: str = DEFAULT_STORE,
    top_k: int = DEFAULT_FINAL_K,
    max_tokens: int = 600,
    selector: Optional[Selector] = None,
) -> Answer:
    """Run the complete RAG flow for one question.

    Args:
        question: the question to answer.
        persist_dir: Chroma persist directory (default: canonical store).
        top_k: number of chunks retrieved for context.
        max_tokens: maximum tokens for the generated answer.
        selector: optional claim->chunk selector callable. Defaults to the
            LLM selector in src.sources_judge. On SourceJudgeError the
            pipeline fails closed (no fallback to broader citations).

    Returns:
        Answer with question, generated answer, claims, retrieved chunks,
        citation-validation status, and validation errors.
    """
    chunks = retrieve_top_k(question, persist_dir=persist_dir, final_k=top_k)
    answer = generate_answer(question, chunks, max_tokens=max_tokens)

    selector = selector or select_supporting_chunks
    try:
        claims = build_citations(
            QUESTION_ID,
            question,
            answer,
            chunks,
            selector=selector,
        )
    except SourceJudgeError as e:
        return Answer(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            citations_valid=False,
            validation_errors=[f"citation selection failed [{e.category}]: {e}"],
        )

    valid, errors = validate_citations(claims, chunks)
    return Answer(
        question=question,
        answer=answer,
        claims=claims,
        retrieved_chunks=chunks,
        citations_valid=valid,
        validation_errors=errors,
    )