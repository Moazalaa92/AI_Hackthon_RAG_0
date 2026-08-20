"""Public FastAPI and Gradio application for the grounded guideline RAG."""

import asyncio
import copy
import sys
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.schemas import (
    AskRequest,
    AskResponse,
    CitationResponse,
    ClaimResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    RatingResponse,
    RatingSignals,
    ReadyResponse,
    RetrievedChunkResponse,
    SafetyResponse,
    VersionResponse,
)
from src.config import (
    MAX_CONCURRENT_REQUESTS,
    QUESTION_CACHE_SIZE,
    RATE_LIMIT_GLOBAL_DAILY,
    RATE_LIMIT_PER_IP_HOUR,
    REQUEST_TIMEOUT_SECONDS,
)
from src.feedback import initialize_feedback_db, log_feedback, log_request
from src.pipeline import DEFAULT_STORE
from src.rating import RatingResult, rate_answer
from src.safety import SafetyResult
from src.safety import query as safety_query
from src.version import VERSION
from src.warmup import warm_up


class QueryTimeout(Exception):
    """Raised when the request queue or pipeline exceeds its timeout."""


class PublicQueryError(Exception):
    """A safe error that can be rendered by either transport surface."""

    def __init__(self, status_code: int, code: str, detail: str, retry_after=None):
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.retry_after = retry_after


class RateLimiter:
    """Small in-process fixed-window limiter for one Space process."""

    def __init__(self, per_ip_hour: int, global_daily: int):
        self.per_ip_hour = per_ip_hour
        self.global_daily = global_daily
        self._lock = threading.Lock()
        self._per_ip = defaultdict(deque)
        self._global = deque()

    def check(self, ip: str) -> int | None:
        now = time.time()
        hour_ago = now - 3600
        day_ago = now - 86400
        with self._lock:
            while self._global and self._global[0] <= day_ago:
                self._global.popleft()
            bucket = self._per_ip[ip]
            while bucket and bucket[0] <= hour_ago:
                bucket.popleft()
            if self.global_daily and len(self._global) >= self.global_daily:
                return max(1, int(self._global[0] + 86400 - now))
            if self.per_ip_hour and len(bucket) >= self.per_ip_hour:
                return max(1, int(bucket[0] + 3600 - now))
            bucket.append(now)
            self._global.append(now)
        return None


_pipeline_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)
_rate_limiter = RateLimiter(RATE_LIMIT_PER_IP_HOUR, RATE_LIMIT_GLOBAL_DAILY)
_cache_lock = threading.RLock()
_question_cache = OrderedDict()


def _normalise_question(question: str) -> str:
    return " ".join(question.casefold().split())


def _client_ip(request) -> str:
    headers = getattr(request, "headers", {}) or {}
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        first_hop = forwarded.split(",", 1)[0].strip()
        if first_hop:
            return first_hop
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "unknown"


def _query_sync(question: str):
    acquired = _pipeline_semaphore.acquire(timeout=REQUEST_TIMEOUT_SECONDS)
    if not acquired:
        raise QueryTimeout
    try:
        result, safety = safety_query(
            question,
            persist_dir=DEFAULT_STORE,
            top_k=10,
        )
        return result, safety, rate_answer(result, safety)
    finally:
        _pipeline_semaphore.release()


async def _query_with_timeout(question: str):
    return await asyncio.wait_for(
        asyncio.to_thread(_query_sync, question),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _cached_query(question: str):
    key = _normalise_question(question)
    with _cache_lock:
        cached = _question_cache.get(key)
        if cached is not None:
            _question_cache.move_to_end(key)
            return copy.deepcopy(cached)
    return None


def _store_cached_query(question: str, value) -> None:
    key = _normalise_question(question)
    with _cache_lock:
        _question_cache[key] = copy.deepcopy(value)
        _question_cache.move_to_end(key)
        while len(_question_cache) > QUESTION_CACHE_SIZE:
            _question_cache.popitem(last=False)


def _to_claim_response(claim) -> ClaimResponse:
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


def _to_rating_response(rating: RatingResult) -> RatingResponse:
    return RatingResponse(
        band=rating.band,
        signals=RatingSignals(**rating.signals),
        reasons=rating.reasons,
    )


def _to_response(
    request_id: str,
    result,
    safety: SafetyResult,
    rating: RatingResult,
) -> AskResponse:
    return AskResponse(
        request_id=request_id,
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
        rating=_to_rating_response(rating),
    )


async def _guarded_query(question: str, ip: str) -> AskResponse:
    """Run one question through the shared API/UI protection path."""
    request_id = uuid.uuid4().hex
    retry_after = _rate_limiter.check(ip)
    if retry_after is not None:
        raise PublicQueryError(
            429,
            "rate_limited",
            "Question limit reached. Please try again later.",
            retry_after,
        )
    if not getattr(app.state, "ready", False):
        raise PublicQueryError(
            503,
            "not_ready",
            "The service is still warming up.",
        )

    started = time.perf_counter()
    error_code = None
    try:
        cached = _cached_query(question)
        if cached is None:
            cached = await _query_with_timeout(question)
            _store_cached_query(question, cached)
        result, safety, rating = cached
        response = _to_response(request_id, result, safety, rating)
    except asyncio.TimeoutError as exc:
        error_code = "timeout"
        raise PublicQueryError(
            504,
            "timeout",
            "The question took too long to process. Please try again.",
        ) from exc
    except QueryTimeout as exc:
        error_code = "timeout"
        raise PublicQueryError(
            504,
            "timeout",
            "The service is busy. Please try again.",
        ) from exc
    except Exception as exc:
        error_code = "internal_error"
        raise PublicQueryError(
            500,
            "internal_error",
            "The question could not be answered. Please try again later.",
        ) from exc
    finally:
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            if "response" in locals():
                log_request(
                    request_id=request_id,
                    ip=ip,
                    question=question,
                    answer=response.answer,
                    band=response.rating.band,
                    signals=response.rating.signals.model_dump(),
                    latency_ms=latency_ms,
                )
            else:
                log_request(
                    request_id=request_id,
                    ip=ip,
                    question=question,
                    answer="",
                    band="error",
                    signals={},
                    latency_ms=latency_ms,
                    error=error_code or "request_failed",
                )
        except Exception:  # noqa: BLE001, S110
            pass
    return response


def _error(status_code: int, code: str, detail: str, retry_after: int | None = None):
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(detail=detail, code=code).model_dump(),
        headers=headers,
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.ready = False
    initialize_feedback_db()
    try:
        warm_up(DEFAULT_STORE)
    except Exception:  # noqa: BLE001
        application.state.warmup_error = "warm-up failed"
    else:
        application.state.ready = True
    yield


app = FastAPI(
    title="NICE NG217 Grounded RAG",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ErrorResponse}},
)
def ready(request: Request):
    if not getattr(request.app.state, "ready", False):
        return _error(503, "not_ready", "The service is still warming up.")
    return ReadyResponse(status="ready")


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(**VERSION)


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def ask(request: Request, body: AskRequest):
    try:
        return await _guarded_query(body.question, _client_ip(request))
    except PublicQueryError as error:
        return _error(
            error.status_code,
            error.code,
            error.detail,
            error.retry_after,
        )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(body: FeedbackRequest) -> FeedbackResponse:
    try:
        log_feedback(
            request_id=body.request_id,
            helpful=body.helpful,
            reason=body.reason,
        )
    except Exception:  # noqa: BLE001
        return _error(
            500,
            "feedback_error",
            "Feedback could not be recorded. Please try again later.",
        )
    return FeedbackResponse(status="recorded")


def _citation_markdown(response: AskResponse) -> str:
    lines = []
    for claim in response.claims:
        for citation in claim.citations:
            page = citation.page_label or str(citation.page or "?")
            lines.append(f"**Page {page}**\n\n> {citation.supporting_text}")
    return "\n\n---\n\n".join(lines) or "No citations were returned."


_UI_CORPUS_MEASUREMENT = (
    "Corpus-level measurement on the 44-question judged eval set "
    "(not this answer): 75.0% judged strictly correct "
    "(95% CI 60.6–85.4%), 20.5% partially correct, 4.5% incorrect, "
    "and 100% grounded in retrieved text."
)


def _ui_submit(question: str, request: gr.Request):
    if not question or not question.strip():
        return "", "", "", "", "", ""
    try:
        response = asyncio.run(
            _guarded_query(question.strip(), _client_ip(request))
        )
    except PublicQueryError as error:
        return error.detail, "", "", "", "", ""
    except Exception:  # noqa: BLE001
        return "The question could not be answered. Please try again later.", "", "", "", "", ""
    if response.rating.band == "refused":
        band = ""
        explanation = (
            "No evidence-support rating applies because this request was refused."
        )
    elif response.rating.band == "low":
        band = "Citations incomplete — treat with caution."
        explanation = (
            "Some claims are not cited to a guideline page. "
            "This describes citation completeness, not accuracy."
        )
    else:
        band = "Supported: every claim is cited to a guideline page."
        explanation = (
            "This band describes support from the retrieved evidence, not accuracy."
        )
    return (
        response.answer,
        band,
        explanation,
        _citation_markdown(response),
        response.request_id,
        "",
    )


def _ui_feedback(request_id: str, helpful: bool) -> str:
    if not request_id:
        return "Ask a question first."
    feedback(FeedbackRequest(request_id=request_id, helpful=helpful))
    return "Thanks for the feedback."


def build_demo():
    with gr.Blocks(title="NICE NG217 Grounded RAG") as demo:
        gr.Markdown(
            "# NICE NG217 grounded answers\n"
            "Informational only, based on NICE NG217, not individualised medical advice.\n\n"
            f"{_UI_CORPUS_MEASUREMENT}"
        )
        question = gr.Textbox(
            label="Question",
            placeholder="Ask a general question about the NICE epilepsy guideline",
        )
        gr.Examples(
            examples=[
                "What is the first-line treatment for focal seizures?",
                "What information should be given about SUDEP?",
                "When should sodium valproate be avoided?",
            ],
            inputs=question,
        )
        submit = gr.Button("Ask", variant="primary")
        answer = gr.Markdown(label="Answer")
        band = gr.Markdown(label="Evidence support")
        band_explanation = gr.Markdown()
        with gr.Accordion("Citations", open=False):
            citations = gr.Markdown()
        request_id = gr.State("")
        with gr.Row():
            helpful = gr.Button("👍 Helpful")
            unhelpful = gr.Button("👎 Not helpful")
        feedback_status = gr.Markdown()
        outputs = [answer, band, band_explanation, citations, request_id, feedback_status]
        submit.click(_ui_submit, inputs=question, outputs=outputs)
        helpful.click(
            lambda request_id: _ui_feedback(request_id, True),
            inputs=request_id,
            outputs=feedback_status,
        )
        unhelpful.click(
            lambda request_id: _ui_feedback(request_id, False),
            inputs=request_id,
            outputs=feedback_status,
        )
    return demo


demo = build_demo()
gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
