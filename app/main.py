"""FastAPI HTTP layer (Phase 10.5): a thin API around src.pipeline.

This module ONLY transports requests and responses. All RAG business logic
lives in src.pipeline.query(); the routes never re-implement retrieval,
generation, or citation logic.

    Client -> FastAPI -> src.pipeline.query() -> structured JSON response

Endpoints:
    GET  /health   -> service status
    POST /ask      -> run one stateless RAG query and return the Answer

The API is intentionally minimal and stateless: no auth, streaming,
websockets, background jobs, databases, or conversation memory in this phase.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException

from app.schemas import (
    AskRequest,
    AskResponse,
    CitationResponse,
    ClaimResponse,
    ErrorResponse,
    HealthResponse,
    RetrievedChunkResponse,
    SafetyResponse,
)
from src.safety import query as safety_query

app = FastAPI(title="RAG Guideline API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a simple machine-readable health status."""
    return HealthResponse(status="ok")


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={500: {"model": ErrorResponse}},
)
def ask(request: AskRequest) -> AskResponse:
    """Run one stateless, safety-checked RAG query and return the structured answer.

    The safety layer (src.safety.query) classifies the question first:
      - PATIENT_SPECIFIC -> returns a safe refusal (pipeline not executed)
      - otherwise -> runs the frozen pipeline, then interprets the result
    Citation-selection failure is NOT an HTTP error: the pipeline preserves
    the generated answer and reports citations_valid=False. Only unexpected
    pipeline failures (e.g. retrieval/store or API errors) return HTTP 500
    with a safe, user-readable message.
    """
    try:
        result, safety = safety_query(
            request.question,
            persist_dir="data/chroma_db/experiments/large_800_100",
            top_k=10,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The question could not be answered. Please try again later.",
        ) from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        claims=[_to_claim_response(claim) for claim in result.claims],
        retrieved_chunks=[
            _to_chunk_response(chunk) for chunk in result.retrieved_chunks
        ],
        citations_valid=result.citations_valid,
        validation_errors=result.validation_errors,
        safety=SafetyResponse(
            classification=safety.classification,
            reason=safety.reason,
            safe_to_answer=safety.safe_to_answer,
        ),
    )


def _to_claim_response(claim) -> ClaimResponse:
    """Map a pipeline claim dict to its API schema (no internal details)."""
    return ClaimResponse(
        claim_id=claim.get("claim_id", ""),
        claim_text=claim.get("claim_text", ""),
        citations=[
            CitationResponse(
                citation_id=c.get("citation_id", ""),
                chunk_id=c.get("chunk_id"),
                document_id=c.get("document_id"),
                source=c.get("source"),
                page=c.get("page"),
                page_label=c.get("page_label"),
                reranker_rank=c.get("reranker_rank"),
                reranker_score=c.get("reranker_score"),
                fusion_score=c.get("fusion_score"),
                rank=c.get("rank"),
                retrieved_by=c.get("retrieved_by"),
                supporting_text=c.get("supporting_text", ""),
            )
            for c in claim.get("citations", [])
        ],
    )


def _to_chunk_response(chunk) -> RetrievedChunkResponse:
    """Map a pipeline chunk record to its API schema (traceable metadata only)."""
    return RetrievedChunkResponse(
        chunk_id=chunk.get("chunk_id"),
        document_id=chunk.get("document_id"),
        source=chunk.get("source"),
        page=chunk.get("page"),
        page_label=chunk.get("page_label"),
        section=chunk.get("section"),
        reranker_rank=chunk.get("reranker_rank"),
        reranker_score=chunk.get("reranker_score"),
        fusion_score=chunk.get("fusion_score"),
        rank=chunk.get("rank"),
        retrieved_by=chunk.get("retrieved_by"),
        chunk_text=chunk.get("chunk_text", ""),
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000, log_level="info")
