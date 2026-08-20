"""Pydantic request/response models for the RAG HTTP API (Phase 10.5).

These schemas are the API contract. They mirror the fields produced by
src.pipeline.query() but are defined independently so the transport layer
never leaks pipeline internals or credentials.

Retrieved-chunk metadata is limited to what is useful for debugging, source
traceability, and client display (chunk id, page, source, ranks, scores,
text). API keys / credentials / configuration are never exposed here.
"""

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


class RatingSignals(BaseModel):
    """Observable signals used to assign an evidence-support band."""

    top1_reranker_score: float | None = None
    top1_top3_margin: float | None = None
    page_agreement_top5: int = 0
    dual_retrieval: bool = False
    claim_count: int = 0
    cited_claim_count: int = 0
    claim_citation_coverage: float = 0.0
    citations_valid: bool | None = None
    safety_classification: str


class RatingResponse(BaseModel):
    """Deterministic support band and its explanations."""

    band: str
    signals: RatingSignals
    reasons: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str


class CitationResponse(BaseModel):
    """A single citation: a claim tied to an exact retrieved chunk."""

    citation_id: str
    chunk_id: int
    document_id: str | None = None
    source: str | None = None
    page: int | None = None
    page_label: str | None = None
    reranker_rank: int | None = None
    reranker_score: float | None = None
    fusion_score: float | None = None
    rank: int | None = None
    retrieved_by: str | None = None
    supporting_text: str


class ClaimResponse(BaseModel):
    """A claim from the answer and its supporting citations."""

    claim_id: str
    claim_text: str
    citations: list[CitationResponse] = Field(default_factory=list)


class RetrievedChunkResponse(BaseModel):
    """A retrieved chunk's metadata, for debugging and traceability."""

    chunk_id: int
    document_id: str | None = None
    source: str | None = None
    page: int | None = None
    page_label: str | None = None
    section: str | None = None
    reranker_rank: int | None = None
    reranker_score: float | None = None
    fusion_score: float | None = None
    rank: int | None = None
    retrieved_by: str | None = None
    chunk_text: str


class AskResponse(BaseModel):
    """POST /ask response: the structured result of one RAG query.

    citations_valid is False (not an HTTP error) when citation selection
    failed but the answer was still produced and preserved — the pipeline
    fails closed rather than inventing or broadening citations.
    """

    request_id: str
    question: str
    answer: str
    claims: list[ClaimResponse] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkResponse] = Field(default_factory=list)
    citations_valid: bool | None = None
    validation_errors: list[str] = Field(default_factory=list)
    safety: SafetyResponse
    rating: RatingResponse


class ErrorResponse(BaseModel):
    """Safe, user-readable error body (no stack traces / internals)."""

    detail: str
    code: str = "request_error"


class FeedbackRequest(BaseModel):
    """POST /feedback request body."""

    request_id: str = Field(..., min_length=1, max_length=128)
    helpful: bool
    reason: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    """POST /feedback response."""

    status: str


class ReadyResponse(BaseModel):
    """GET /ready response."""

    status: str


class VersionResponse(BaseModel):
    """GET /version response."""

    corpus: str
    corpus_version: str
    corpus_date: str
    embedding_model: str
    reranker_model: str
    generation_model: str
    prompt_version: str
    rating_rule_version: str
