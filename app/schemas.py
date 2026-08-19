"""Pydantic request/response models for the RAG HTTP API (Phase 10.5).

These schemas are the API contract. They mirror the fields produced by
src.pipeline.query() but are defined independently so the transport layer
never leaks pipeline internals or credentials.

Retrieved-chunk metadata is limited to what is useful for debugging, source
traceability, and client display (chunk id, page, source, ranks, scores,
text). API keys / credentials / configuration are never exposed here.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

QUESTION_MAX_LENGTH = 1000


class AskRequest(BaseModel):
    """POST /ask request body."""

    question: str = Field(..., min_length=1, max_length=QUESTION_MAX_LENGTH)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped


class SafetyResponse(BaseModel):
    """Safety/intent classification for one question (Phase 11).

    `safe_to_answer` means "allowed to proceed under the current safety policy".
    It does NOT mean clinically safe, medically validated, or guaranteed
    correct.
    """

    classification: str
    reason: str
    safe_to_answer: bool


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str


class CitationResponse(BaseModel):
    """A single citation: a claim tied to an exact retrieved chunk."""

    citation_id: str
    chunk_id: int
    document_id: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None
    page_label: Optional[str] = None
    reranker_rank: Optional[int] = None
    reranker_score: Optional[float] = None
    fusion_score: Optional[float] = None
    rank: Optional[int] = None
    retrieved_by: Optional[str] = None
    supporting_text: str


class ClaimResponse(BaseModel):
    """A claim from the answer and its supporting citations."""

    claim_id: str
    claim_text: str
    citations: List[CitationResponse] = Field(default_factory=list)


class RetrievedChunkResponse(BaseModel):
    """A retrieved chunk's metadata, for debugging and traceability."""

    chunk_id: int
    document_id: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None
    page_label: Optional[str] = None
    section: Optional[str] = None
    reranker_rank: Optional[int] = None
    reranker_score: Optional[float] = None
    fusion_score: Optional[float] = None
    rank: Optional[int] = None
    retrieved_by: Optional[str] = None
    chunk_text: str


class AskResponse(BaseModel):
    """POST /ask response: the structured result of one RAG query.

    citations_valid is False (not an HTTP error) when citation selection
    failed but the answer was still produced and preserved — the pipeline
    fails closed rather than inventing or broadening citations.
    """

    question: str
    answer: str
    claims: List[ClaimResponse] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunkResponse] = Field(default_factory=list)
    citations_valid: Optional[bool] = None
    validation_errors: List[str] = Field(default_factory=list)
    safety: SafetyResponse


class ErrorResponse(BaseModel):
    """Safe, user-readable error body (no stack traces / internals)."""

    detail: str