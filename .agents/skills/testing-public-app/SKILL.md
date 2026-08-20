---
name: testing-public-app
description: How to run and end-to-end test the public NICE NG217 RAG app (FastAPI + Gradio on port 7860), including rate limits, safety refusals, and testing without an LLM API key.
---

# Testing the public shared app (app/main.py)

## Run it
```
cd <repo>   # venv at .venv
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 7860
```
- Gradio UI at `http://localhost:7860/`, API at `POST /ask`, plus `/health`, `/ready`, `/version`, `POST /feedback`.
- Startup warm-up (MiniLM embeddings, Chroma `data/chroma_db/experiments/large_800_100`, cross-encoder) takes ~10-15 s.
  Uvicorn does **not** accept connections until the lifespan warm-up finishes, so the pre-warm-up `/ready` 503 is
  **not reachable over HTTP** (you get connection-refused instead). To exercise 503, drive the ASGI app in-process:
  ```python
  from fastapi.testclient import TestClient; from app.main import app  # PYTHONPATH=<repo root>
  c = TestClient(app)          # lifespan not run -> app.state.ready unset
  print(c.get("/ready").status_code)   # 503, {"code":"not_ready"}
  app.state.ready = True; print(c.get("/ready").status_code)  # 200
  ```
  Run it with `PYTHONPATH=<repo root>` or the `app` package will not import.

## Testing without LLM_API_KEY
Without `LLM_API_KEY` (there is no `.env` by default), every question that passes the safety gate hits the generation
error path and the UI renders exactly `The question could not be answered. Please try again later.` (`/ask` -> 500
`internal_error`). This is expected, not a bug. Testable without a key: disclaimer/corpus line, patient-specific
refusal, error rendering, rate limits, cache, feedback, `/health` `/ready` `/version`, adversarial input.
NOT testable: successful answers, support bands (high/medium/low), citation accordion contents,
INSUFFICIENT_EVIDENCE classification, layer-2 LLM intent classifier (`SAFETY_INTENT_LLM_ENABLED=1`).
Devin Secrets Needed: `LLM_API_KEY` (plus optional `LLM_BASE_URL`, `LLM_MODEL`) to test generation paths.

## Env knobs that make testing fast (src/config.py)
- `RATE_LIMIT_PER_IP_HOUR` (default 10) — lower it (e.g. 6) to reach "Question limit reached. Please try again later."
  in a few UI clicks. `RATE_LIMIT_GLOBAL_DAILY` (1000), `QUESTION_CACHE_SIZE`, `MAX_CONCURRENT_REQUESTS`,
  `REQUEST_TIMEOUT_SECONDS`, `FEEDBACK_DB_PATH`, `FEEDBACK_IP_SALT`.
- The limiter is in-process and per client IP; the browser at localhost and shell `curl` share the same 127.0.0.1
  bucket, so you can exhaust the budget in the UI and then confirm `POST /ask` returns 429 + `Retry-After`.
- Rate limiting is checked BEFORE the cache, so cached repeats still consume budget. Restarting the process resets
  both limiter and cache.

## Budget accounting gotchas
- Empty / whitespace-only submits return early in the UI and do **not** consume budget (`app/main.py:_ui_submit`).
- Only successful responses are cached; error responses are not. With no API key, that means **only refusals are
  cacheable** — prove cache hits with a patient-specific question asked twice and compare `latency_ms` in
  `data/feedback.sqlite3`.

## Verifying logging / feedback
```
.venv/bin/python -c "import sqlite3;c=sqlite3.connect('data/feedback.sqlite3');
print(c.execute('select timestamp,band,latency_ms,error,substr(question,1,40) from requests').fetchall());
print(c.execute('select request_id,helpful from feedback').fetchall())"
```

## Gradio UI gotcha
The question box is a growing textarea: after typing a long question the "Ask" button moves down the page. Always
take a screenshot and re-locate the Ask button before clicking, otherwise the click lands in the textarea and the
previous (stale) answer stays on screen and can be mistaken for a fresh result.

## Safety gate behaviour worth knowing
The layer-1 gate (`src/safety.py`) is a regex over personal references — any `my/i/me/we/our/us`, age vignette
("a 7-year-old"), "this patient", or third-person clinical narrative -> PATIENT_SPECIFIC refusal, pipeline not
called, band `refused`, UI shows no support label plus "No evidence-support rating applies because this request was
refused." Note "tell **me** ..." phrasing therefore refuses, so prompt-injection probes may be refused by the gate
rather than reaching generation.
