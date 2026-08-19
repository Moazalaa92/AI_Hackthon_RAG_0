# Hackathon Lectures — Grounded RAG over a NICE Epilepsy Guideline (NG217)

A **retrieval-augmented generation (RAG)** question-answering system over the clinical
guideline *"Epilepsies in children, young people and adults"* (**NICE NG217**,
`data/pdfs/Project_pdf.pdf`, 161 pages).

This is **not merely a chatbot.** The core objective is an **evidence chain**:

```text
Question
  → find relevant evidence
  → generate an evidence-grounded answer
  → attach traceable citations
  → validate the evidence chain
  → expose the result through an API
```

Every answer claim is traceable back to the **exact guideline chunk, page, and source
text** that supports it — or the system refuses rather than invent.

---

## Project status

| Area | Status |
|---|---|
| **Retrieval** | **DONE / FROZEN** (current iteration) |
| **Generation** | **DONE / FROZEN** (current iteration) |
| **Citations / traceability** | **IMPLEMENTED / VALIDATED** |
| **End-to-end pipeline** | **DONE** (Phase 10 — `src/pipeline.py`) |
| **FastAPI API** | **DONE** (Phase 10.5 — `app/main.py`) |
| **Safety / guardrails** | **IN PROGRESS** (Phase 11 — being designed & evaluated) |
| **Confidence / observability** | NEXT (Phase 12) |
| **End-to-end evaluation** | NEXT (Phase 13) |
| **UI / client** | OPTIONAL / LATER (Phase 14, Gradio is *not* the production backend) |
| **Conversation memory** | FUTURE (the API is intentionally stateless) |

The full phase-by-phase plan and authoritative phase-status table live in `PLAN.md`.

---

## What the system does

Given a question about the NG217 epilepsy guideline:

1. **Retrieves** the most relevant evidence chunks from the guideline (hybrid dense + BM25,
   cross-encoder reranked).
2. **Generates** a concise, grounded answer using **only** that retrieved context.
3. **Splits** the answer into claims and attaches a **citation** to each claim, pointing at
   the exact supporting chunk, page, and source text.
4. **Validates** every citation deterministically (cited chunk actually retrieved, text and
   page metadata match).
5. **Returns** a structured `Answer` through the FastAPI API.

The whole flow runs in one call:

```text
pipeline.query(question) → Answer
```

---

## Architecture (canonical, frozen)

```text
Document ingestion (NICE NG217 PDF)
      ↓
Chunking: 800 / 100 (characters / overlap)
      ↓
Embeddings: sentence-transformers/all-MiniLM-L6-v2
      ↓
  ┌─────────────────────────────┐
  │                             │
  ↓                             ↓
Dense Top-20                BM25 Top-20
(Chroma, L2)                (lexical)
  │                             │
  └──────────────┬──────────────┘
                 ↓
        Candidate fusion (RRF, k=60)
        — builds the candidate union + fusion scores
                 ↓
        Cross-Encoder reranking
        cross-encoder/ms-marco-MiniLM-L-6-v2
        — the FINAL ranking signal
                 ↓
           Final Top-10
                 ↓
        Generation V2 (deepseek/deepseek-v4-flash, temp 0)
                 ↓
        Claims
                 ↓
        Citation mapping
                 ↓
        Citation validation
                 ↓
        Structured Answer
                 ↓
        FastAPI (GET /health, POST /ask)
```

**About RRF:** RRF fuses dense + BM25 into a candidate pool and provides the *fusion
score* metadata. It is **not** the final ranking: RRF capped single-retriever chunks at
~1/(k+1) and hurt recall of BM25-only evidence (the Q09 case). The final Top-10 is ranked
purely by the cross-encoder. The historical RRF artifacts are preserved for reproducibility.

### Configuration summary

| Setting | Value |
|---|---|
| Chunk size / overlap | `800` / `100` characters |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Dense retrieval | Chroma, L2 distance, Top-20 |
| BM25 | lexical, built from the same stored chunks, Top-20 |
| Candidate fusion | RRF, k=60 (candidate pool only) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Final retrieval | Top-10 |
| Generation model | `deepseek/deepseek-v4-flash`, temperature 0 |
| Canonical store | `data/chroma_db/experiments/large_800_100` |

Score semantics: **Chroma L2** — lower = more similar. **Cross-encoder** — higher = more
relevant (the final ranking signal).

---

## Retrieval results (verified, frozen architecture)

Relevance labels come from an **LLM-as-a-Judge** (see [Evaluation methodology](#evaluation-methodology)).
These are **retrieval-quality numbers, not answer accuracy.**

### Benchmark — 20 questions (`evaluation/dataset.json`)

```text
P@1   = 0.850
P@3   = 0.517
P@5   = 0.400
P@10  = 0.230
MRR   = 0.879
Hit@5 = 0.95
```

### Independent holdout — 27 questions (`evaluation/holdout_dataset_v1.json`)

```text
P@1   = 0.815
P@3   = 0.580
P@5   = 0.452
P@10  = 0.311
MRR   = 0.870
```

### Candidate recall

```text
Benchmark: 19/19 answerable questions = 100%
Holdout:   25/25 answerable questions = 100%
```

**Interpretation.** The system is **strong at finding the relevant evidence** (P@1 ≈ 0.85,
MRR ≈ 0.88, candidate recall ≈ 100%). The main remaining retrieval limitation is
**top-3 / top-5 precision density** — relevant evidence is almost always retrieved, but
occasionally a distractor outranks it just below the top.

> **P@K and Hit@K are retrieval metrics — NOT answer accuracy.** They measure whether the
> right evidence chunks surface, not whether a correct final answer is produced. The higher
> holdout P@3 is **not** a system improvement; it is the *same frozen architecture* on a
> different dataset, confirming the design generalizes.

### Ranking optimization (opt-in, Phase 13.1)

The final ranking stage was swept offline over the **already-labeled** candidate pools
(`scripts/sweep_rankers.py`, no judge calls, candidate generation unchanged). All numbers
below use the candidate-pool labels for every strategy, so they are directly comparable to
each other (the single-model row is 0.5333 here vs 0.5167 in the frozen artifact purely
because of judge label variance between runs).

| Ranker | Benchmark P@3 | Holdout P@3 |
|---|---:|---:|
| RRF (no reranker) | 0.4167 | 0.5062 |
| `ms-marco-MiniLM-L-6-v2` (default) | 0.5333 | 0.5802 |
| `BAAI/bge-reranker-v2-m3` | 0.5500 | 0.5679 |
| **ensemble(L-6 + bge-v2-m3) + 0.1·BM25** | **0.5833** | **0.5926** |
| Achievable ceiling given the labels | 0.7333 | 0.8148 |

The ensemble averages the two cross-encoders' **per-question z-standardized** scores (raw
cross-encoder logits are on incomparable scales) and adds a small z-standardized BM25 term.
It is **opt-in**, not the default: the gain is +0.05 / +0.01 P@3 on 20 / 27 questions and
the bootstrap 95% CI includes zero on both sets, so it is the best observed configuration,
not a proven one. Run it with:

```bash
python scripts/run_rerank_evaluation.py \
  --candidates evaluation/results/hybrid_800_100_candidates.jsonl \
  --name ensemble_hybrid_800_100 \
  --models cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-v2-m3 \
  --bm25-weight 0.1
```

**A mean P@3 of 0.85 is unreachable on these datasets.** P@3 divides by 3 regardless of how
much relevant evidence exists: 4 benchmark questions have a single relevant chunk in the
entire labeled pool (P@3 ≤ 0.333 each) and Q20 is a deliberate unanswerable question scored
0. A perfect ranker therefore scores 0.7333 on the benchmark and 0.8148 on the holdout.
Raising that ceiling requires changing candidate generation and re-judging the pool, not
tuning the ranker — Hit@3 (0.95 / 0.9259) and candidate recall (100%) are the metrics that
are actually near saturation.

This differs from the rejected "Idea B — second lightweight ranking layer"
(`evaluation/experiments/adaptive_context_vs_second_ranker/REPORT.md`), which reordered the
**frozen Top-10** from a **single** cross-encoder with a secondary signal. The gain here
comes from a **second cross-encoder** scoring the **full candidate union**, so chunks below
rank 10 can still enter the Top-3; the BM25 term alone reproduces Idea B's null result.

## Pixel RAG (visual page retrieval, opt-in)

Pixel RAG is an additive experiment that renders each PDF page as an image and retrieves
pages with ColSmol's multi-vector late-interaction MaxSim scorer. The extracted-text
retrieval and ranking path above remains frozen. The visual dependencies are intentionally
separate from `requirements.txt`:

```bash
python3 -m venv .venv-visual
.venv-visual/bin/pip install -r requirements-visual.txt
```

Build the 150-DPI page index once. On the reference 2-vCPU CPU machine this takes
approximately **44 minutes** and is resumable with a checkpoint every 20 pages:

```bash
.venv-visual/bin/python scripts/build_page_index.py \
    --pdf data/pdfs/Project_pdf.pdf \
    --dpi 150 \
    --model vidore/colSmol-256M \
    --out data/page_index/colsmol_150dpi.npz
```

Run the two page-retrieval evaluations:

```bash
.venv-visual/bin/python scripts/run_visual_evaluation.py \
    --dataset evaluation/dataset.json \
    --index data/page_index/colsmol_150dpi.npz \
    --name visual_colsmol_benchmark
.venv-visual/bin/python scripts/run_visual_evaluation.py \
    --dataset evaluation/holdout_dataset_v1.json \
    --index data/page_index/colsmol_150dpi.npz \
    --name visual_colsmol_holdout
```

Page metrics are offline and use either annotated `expected_pages` or pages marked
`relevant` in the existing labeled pools:

```bash
.venv-visual/bin/python scripts/page_metrics.py \
    --input evaluation/results/visual_colsmol_benchmark_visual_pages.jsonl \
    --dataset evaluation/dataset.json --ground-truth annotated \
    --name visual_colsmol_benchmark_annotated
.venv-visual/bin/python scripts/page_metrics.py \
    --input evaluation/results/visual_colsmol_benchmark_visual_pages.jsonl \
    --dataset evaluation/dataset.json --ground-truth judged \
    --pool evaluation/results/hybrid_800_100_candidates_labeled.jsonl \
    --name visual_colsmol_benchmark_judged
```

The visual-page prior can also be swept over the frozen labeled chunk pools:

```bash
.venv-visual/bin/python scripts/sweep_visual_prior.py \
    evaluation/results/hybrid_800_100_candidates_labeled.jsonl \
    evaluation/results/holdout_v1_hybrid_candidates_labeled.jsonl \
    --dataset evaluation/dataset.json \
    --holdout-dataset evaluation/holdout_dataset_v1.json \
    --index data/page_index/colsmol_150dpi.npz
```

Measured page retrieval results (PageHit@K / PageCoverage@K) are recorded in
`evaluation/metrics/`:

| Dataset / ground truth | Visual Hit@1 | Visual Hit@3 | Visual Hit@5 | Visual Hit@10 | Visual Coverage@3 | Visual Coverage@10 | Text Hit@3 | Text Hit@10 | Text Coverage@3 | Text Coverage@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benchmark / annotated (n=19, 1 excluded) | 0.6842 | 0.8947 | 0.8947 | 0.9474 | 0.8684 | 0.9211 | 0.8947 | 1.0000 | 0.8684 | 1.0000 |
| Benchmark / judged (n=19, 1 excluded) | 0.7368 | 0.9474 | 1.0000 | 1.0000 | 0.7216 | 0.8988 | 1.0000 | 1.0000 | 0.8035 | 0.9649 |
| Holdout / annotated (n=0, 27 excluded) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Holdout / judged (n=25, 2 excluded) | 0.9200 | 1.0000 | 1.0000 | 1.0000 | 0.6610 | 0.8790 | 1.0000 | 1.0000 | 0.6910 | 0.9057 |

The visual-prior sweep reports P@3, P@5, Hit@3, MRR, and the pool ceiling for weights
0, 0.1, 0.25, 0.5, and 1.0, plus visual-only ordering, in
`evaluation/metrics/visual_prior_sweep.json`.

| Dataset | Prior strategy | P@3 | P@5 | Hit@3 | MRR |
|---|---|---:|---:|---:|---:|
| Benchmark | visual only | 0.4667 | 0.3600 | 0.8500 | 0.7239 |
| Benchmark | text CE + 0.0 visual | 0.5333 | 0.3900 | 0.9000 | 0.8792 |
| Benchmark | text CE + 0.1 visual | 0.5333 | 0.3800 | 0.9000 | 0.8792 |
| Benchmark | text CE + 0.25 visual | 0.5500 | 0.3800 | 0.9500 | 0.8833 |
| Benchmark | text CE + 0.5 visual | 0.5167 | 0.3700 | 0.9500 | 0.8833 |
| Benchmark | text CE + 1.0 visual | 0.5333 | 0.4000 | 0.9500 | 0.8833 |
| Holdout | visual only | 0.4691 | 0.4074 | 0.8889 | 0.8426 |
| Holdout | text CE + 0.0 visual | 0.5802 | 0.4519 | 0.9259 | 0.8704 |
| Holdout | text CE + 0.1 visual | 0.5679 | 0.4667 | 0.9259 | 0.8704 |
| Holdout | text CE + 0.25 visual | 0.5679 | 0.4815 | 0.9259 | 0.8704 |
| Holdout | text CE + 0.5 visual | 0.5432 | 0.4815 | 0.9259 | 0.8889 |
| Holdout | text CE + 1.0 visual | 0.5556 | 0.4815 | 0.9259 | 0.8889 |

The labeled-pool ceilings are P@3=0.7333 / P@5=0.5300 on the benchmark and
P@3=0.8148 / P@5=0.6444 on holdout. Visual-only retrieval loses substantially to
the text cross-encoder. Adding the visual prior does not improve benchmark P@3
over text-only, and lowers holdout P@3 at every nonzero weight; it raises P@5 on
holdout at weights 0.1–1.0 but does not justify changing the frozen path.

This corpus is a deliberately difficult fit for visual retrieval: all **161 NG217 pages
are pure linear text**, with **0 embedded images**, **no real tables**, and a median of
**1,694 extracted characters per page**. The frozen text pipeline already reaches
**PageHit@3 = 0.8947** and **PageHit@10 = 1.0000**, so ColPali-style retrieval has little
headroom here. The measured visual result is reported plainly, including if visual
retrieval loses.

---

## Retrieval investigation — an engineering story

The retrieval work followed **"measure → diagnose → experiment → reject unsafe/weak
changes → freeze"**:

1. **Dense retrieval alone was insufficient** — P@3 was low and some relevant evidence
   (e.g. Q09) was missed.
2. **BM25 added complementary lexical retrieval** — it found chunks dense missed.
3. **Hybrid retrieval improved candidate coverage** — the union of both retrievers.
4. **Cross-encoder reranking produced the strongest ranking** — P@1 85%, P@3 51.7%,
   MRR 87.9% (the largest single gain, far above the judge-noise floor).
5. **Holdout validation showed the behavior generalizes** — 27 unseen questions.
6. **Candidate recall is already ≈ 100%** — the evidence is always in the pool; the
   remaining work is ranking it to the top.
7. **Further deterministic ranking heuristics were investigated** — several post-reranker
   ideas to sharpen top precision.
8. **H6-A produced a small safe improvement but was not enough** to justify a change.
9. **Adaptive context and a second lightweight deterministic ranker were tested and
   rejected** — gains were not robust or not worth the added complexity.
10. **An oracle-usefulness experiment demonstrated theoretical ranking headroom** — a
    perfect usefulness signal would help, showing the ceiling exists.
11. **A practical non-oracle usefulness classifier was not yet reliable enough** to be the
    final ranker.
12. **Therefore retrieval optimization is currently frozen** — the cross-encoder hybrid is
    the canonical design for this iteration.

Details: `evaluation/experiment_notes.md` (hypothesis → decision log) and
`evaluation/metrics/comparison.md` (full experiment comparison).

---

## Generation (Phase 8 — Generation V2, frozen)

Retrieval produces evidence; Generation turns it into a natural-language answer.

- **Model:** `deepseek/deepseek-v4-flash` via OpenRouter (`src/generation.py`).
- **Temperature:** `0` (deterministic output).
- **Grounding rule:** answer **only** from the provided context; refuse if the context does
  not contain enough information — never invent.

### Measured results (LLM-as-a-Judge over the 44 answerable questions)

```text
Correct:   33/44 = 75.0%
Grounded:  44/44 = 100%
Complete:  40/44 = 90.9%
Refusal:    3/3  = 100%   (deliberate negatives Q20, H26, H27 correctly refused)
```

What these mean:

- **Correct** — the answer matches the reference answer for the question.
- **Grounded** — every claim in the answer is supported by the retrieved context.
- **Complete** — the answer covers the important components of the question.
- **Appropriate refusal** — when the guideline genuinely lacks the answer (dosing, costs),
  the system says so instead of inventing.

Phrase grounding precisely: **"100% grounded according to the current LLM-as-a-Judge
evaluation."** This is **not** a claim of factual accuracy, zero hallucinations, or
clinical safety.

### Known Generation limitations

- multi-part questions sometimes have omissions
- occasional conflation of a *neighboring* recommendation for a similar question
- occasional over-abstention (refusing when the evidence arguably answers)

Generation V2 is **frozen for the current iteration.**

---

## Citations / traceability (Phase 9)

The citation layer makes each answer claim **auditable**. The concept is simple:

```text
Claim → Citation → Retrieved chunk → Page → Exact source text
```

The system splits the answer into sentence-level claims, pre-filters the question's
retrieved Top-10 chunks by lexical overlap, lets an LLM selector pick the supporting
chunks, then copies every citation field **from the actual retrieved chunk** (chunk id,
page, page label, source, exact supporting text) — nothing is invented. Citations are then
validated deterministically.

### Verified artifacts (47 questions, 92 claims, 130 citations)

```text
Deterministic coverage:        88/88 answerable claims cited = 100%
Deterministic traceability:    47/47 questions valid
                              130/130 citations point to actually retrieved chunks
```

### Historical LLM-judged semantic citation support

```text
109/130 supported        = 90.1%
 10/130 partially_supported = 8.3%
  2/130 unsupported        = 1.7%
```

**Keep these three measurements distinct:**

- **Deterministic coverage** — every answerable claim got a citation (100%).
- **Deterministic traceability** — every citation points at a chunk that was really
  retrieved, with matching text/page/source (100%).
- **LLM-judged semantic support** — whether the cited chunk actually *supports* the claim
  (90.1% supported).

The offline deterministic fallback judge (`judge=deterministic`, used when the LLM API is
unavailable) measures lexical token overlap, **not semantic entailment** — it is **not**
equivalent to the LLM judge. The LLM judge is the more reliable measurement.

**Key lesson from this phase:** *"Relevant does not always mean supporting."* A chunk can
be relevant to the question yet not support a specific claim (the H12 C1 and Q16 C2
cases). Retrieval relevance and citation support are deliberately measured as separate
dimensions.

---

## End-to-end pipeline (Phase 10 — `src/pipeline.py`)

One entry point composes the frozen stages:

```text
pipeline.query(question)
  → retrieve_top_k()        frozen hybrid retrieval + cross-encoder → Top-10
  → generate_answer()       frozen Generation V2
  → build_citations()       claim-level citation layer
  → validate_citations()    deterministic traceability checks
  → Answer
```

The internal `Answer` contract:

```text
Answer:
  question
  answer
  claims            (claim_id, claim_text, citations)
  retrieved_chunks  (frozen Top-10 records with metadata)
  citations_valid   (bool)
  validation_errors (list)
```

**Important design decision — the pipeline fails closed.** If citation selection fails
(e.g. the LLM API is unavailable), the pipeline does **not** silently broaden or invent
citations. It preserves the generated answer, sets `citations_valid = false`, and populates
`validation_errors`. Never invent or widen the evidence.

Run it from the terminal:

```bash
python scripts/ask.py "Which medicine is offered as first-line treatment for absence seizures?"
```

---

## FastAPI API (Phase 10.5 — `app/main.py`)

A thin HTTP layer over `src.pipeline.query()`. The API is intentionally minimal and
**stateless**: no auth, no streaming, no conversation memory. All RAG logic stays in the
pipeline; the routes only transport requests and responses.

```text
Client → FastAPI → RAG pipeline → structured response
```

Start the server:

```bash
uvicorn app.main:app --reload
```

Interactive API docs: `http://127.0.0.1:8000/docs`.

### Endpoints

**`GET /health`** — service status:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok"}
```

**`POST /ask`** — one stateless RAG query:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which medicine is offered as first-line treatment for absence seizures?"}'
```

Request schema: `{"question": "..."}` — required, non-empty, at most 1000 characters.

Response schema (abridged): `question`, `answer`, `claims` (each with `claim_id`,
`claim_text`, `citations` → chunk id, page, source, supporting text), `retrieved_chunks`
(the frozen Top-10 with metadata), `citations_valid`, `validation_errors`.

The API provides:

- **request validation** — invalid `question` → HTTP `422`
- **structured response schemas** — typed Pydantic models (`app/schemas.py`)
- **citation validity reporting** — `citations_valid` + `validation_errors`
- **safe error handling** — unexpected failures → HTTP `500` with a safe, user-readable
  message (no stack traces, no credentials); citation-selection failure is reported as
  `citations_valid: false` (HTTP 200), not a crash

---

## Safety / guardrails (Phase 11 — IN PROGRESS)

**Safety is being designed and evaluated — it is not implemented yet.** Do not claim it is.

The approved direction is a minimal, explicit, explainable three-way policy:

- **NORMAL** guideline question — continue through the frozen RAG pipeline unchanged.
- **PATIENT_SPECIFIC** medical request — must **not** receive individualized medical
  advice; refuse and direct to a qualified professional.
- **INSUFFICIENT_EVIDENCE** — the guideline does not support the requested answer; refuse
  rather than hallucinate. (Note: "retrieved context exists" ≠ "retrieved context supports
  the requested answer.")

Design principles in scope: fail closed (never silently assume a request is safe), keep it
small (no policy engines, no conversation memory, no external safety APIs), and evaluate
the policy with a dedicated safety dataset (false positives *and* false negatives).

Future phases: **Confidence / observability** (Phase 12), **End-to-end evaluation**
(Phase 13), optional **UI / client** (Phase 14).

---

## Presentation story

A 60-second version of the whole project, answers to the questions a reviewer will ask:

1. **What problem are we solving?** Answering questions about a clinical guideline is
   high-stakes: a plausible-sounding wrong answer is worse than no answer. We need answers
   that are **grounded in evidence and auditable** — not a free-form chatbot.
2. **What is the source/document?** The NICE NG217 guideline *"Epilepsies in children,
   young people and adults"* (161-page PDF).
3. **What does the system do?** Question → retrieve the most relevant evidence → generate a
   grounded answer → attach a citation to every claim → validate the evidence chain → serve
   it over an API.
4. **What is the architecture?** Hybrid retrieval (dense + BM25 → cross-encoder → Top-10) →
   grounded generation → claim-level citations → deterministic validation → FastAPI. The
   stages are frozen, independently testable, and orchestrated by `pipeline.query()`.
5. **How did we evaluate retrieval?** 20-question benchmark + 27-question holdout, labeled
   by an LLM-as-a-Judge. We measured P@K, MRR, Hit@K, and candidate recall.
6. **What did we learn?** Dense alone wasn't enough; BM25 added coverage; the hybrid
   candidate pool reached ~100% recall; the cross-encoder was the biggest single win
   (P@1 85%, MRR 88%); further deterministic ranking ideas were tested and rejected because
   they weren't robust. The story is **measure → diagnose → experiment → reject weak
   changes → freeze.**
7. **How well does Generation perform?** 33/44 correct (75%), 44/44 grounded (100% by the
   LLM judge), 40/44 complete (90.9%), and 3/3 deliberate negatives correctly refused.
8. **How do citations/traceability work?** Claim → citation → retrieved chunk → page →
   exact source text. 130/130 citations point at really-retrieved chunks; 88/88 answerable
   claims are covered. Key lesson: *relevant ≠ supporting.*
9. **What does the API provide?** `GET /health` and `POST /ask` — validated requests,
   structured responses, citation-validity reporting, safe errors, stateless.
10. **What are the limitations?** Top-3/top-5 precision density is the retrieval weakness;
    Generation has occasional multi-part omissions, neighbor-recommendation conflation, and
    over-abstention; citation support is 90.1% under the LLM judge; all labels are
    LLM-judged, not clinical validation; safety guardrails are not yet implemented.
11. **What is next?** Safety/guardrails (in progress), then confidence/observability, then
    end-to-end evaluation, then an optional UI.

**The central story: "Build → Measure → Diagnose → Improve → Validate → Integrate."**

---

## Quick start

### 1. Clone and install

```bash
git clone <repository-url>
cd hackathon_lectures
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requires **Python 3.10+**. First run downloads the embedding and cross-encoder models from
Hugging Face.

### 2. Configure

```bash
cp .env.example .env
```

`.env` is gitignored — never commit it.

| Variable | Required for | Notes |
|---|---|---|
| `LLM_API_KEY` | Labeling / Generation | OpenRouter API key. **Do not commit.** |
| `LLM_BASE_URL` | Labeling / Generation | e.g. `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | Labeling / Generation | e.g. `deepseek/deepseek-v4-flash` |
| `DATA_DIR` / `CHROMA_DB_DIR` / `PDF_DIR` | Optional | safe defaults (`data`, `data/chroma_db`, `data/pdfs`) |

**Retrieval, ingestion, and metrics need no API key** (local Hugging Face models). Only
LLM-based labeling, Generation, and citation selection need the key.

### 3. Build the vector store

```bash
python scripts/ingest.py --chunk-size 800 --chunk-overlap 100 \
    --persist-dir data/chroma_db/experiments/large_800_100
```

Default `python scripts/ingest.py` builds the baseline store (500/50). Experiment stores
live in `data/chroma_db/experiments/<config>/` and are gitignored (rebuildable).

### 4. Ask a question

```bash
python scripts/ask.py "Which medicine is offered as first-line treatment for absence seizures?"
```

Or through the API (see [FastAPI API](#fastapi-api-phase-105--appmainpy)).

---

## Repository structure

```text
hackathon_lectures/
├── README.md                 ← this file
├── PLAN.md                   ← full plan: phase status, architecture, deviations
├── requirements.txt          ← pinned dependencies
├── requirements-visual.txt   ← opt-in Pixel RAG dependencies
├── .env.example              ← config template (copy to .env)
├── app.py                    ← legacy demo (no LLM) — NOT canonical
│
├── app/                      ← FastAPI layer (Phase 10.5)
│   ├── main.py               ← GET /health, POST /ask (thin routes)
│   └── schemas.py            ← Pydantic request/response models
│
├── src/                      ← core modules
│   ├── config.py             ← reads .env, path/model defaults
│   ├── ingestion.py          ← PDF → pages
│   ├── chunking.py           ← pages → chunks (800/100)
│   ├── embeddings.py         ← all-MiniLM-L6-v2
│   ├── vectorstore.py        ← Chroma persistence + query
│   ├── retrieval.py          ← dense-only baseline path
│   ├── hybrid_retrieval.py   ← dense + BM25 candidate union (fusion/RRF scores)
│   ├── reranking.py          ← cross-encoder reranker (final ranking)
│   ├── generation.py         ← context + question → grounded answer (Phase 8)
│   ├── generation_judge.py   ← correctness/completeness/grounding judge
│   ├── sources.py            ← claim splitting + deterministic citation build (Phase 9)
│   ├── sources_judge.py      ← LLM support judge + deterministic fallback (Phase 9)
│   ├── pipeline.py           ← query() → Answer (Phase 10 orchestration)
│   ├── evaluation.py         ← run fixed dataset retrieval → JSONL
│   ├── metrics.py            ← Precision@K, Hit@K, Top-K analysis
│   ├── judge.py              ← LLM relevance judge (retrieval labels)
│   ├── page_images.py        ← deterministic PDF page rendering
│   └── visual_retrieval.py   ← opt-in ColSmol page retrieval
│
├── scripts/                  ← command-line tools
│   ├── ask.py                ← single-question end-to-end CLI (Phase 8)
│   ├── run_citations.py      ← build claim-level citations (Phase 9)
│   ├── evaluate_citations.py ← judge citation support → metrics (Phase 9)
│   ├── evaluate_generation.py / run_generation_evaluation.py ← Phase 8 eval
│   ├── run_evaluation.py     ← dense-only Top-K on the eval dataset
│   ├── run_hybrid_evaluation.py ← dense+BM25 candidate pool
│   ├── run_rerank_evaluation.py ← cross-encoder rerank → final Top-10
│   ├── sweep_rankers.py      ← offline ranker comparison over labeled pools (no LLM)
│   ├── build_page_index.py / run_visual_evaluation.py ← Pixel RAG CLIs
│   ├── page_metrics.py / sweep_visual_prior.py ← offline visual metrics
│   ├── label_results.py      ← LLM-judge labels for a results JSONL
│   ├── apply_pool_labels.py  ← reuse pool labels for a reordering experiment (no LLM)
│   ├── metrics.py / analyze_topk.py / candidate_recall_analysis.py / rerank_analysis.py
│   ├── ingest.py / retrieve.py / embed_demo.py
│   └── inspect_*.py          ← inspect documents / chunks / vectors / results
│
├── evaluation/               ← frozen evaluation artifacts
│   ├── dataset.json          ← canonical 20-question benchmark
│   ├── holdout_dataset_v1.json ← independent 27-question holdout
│   ├── generation_dataset_v1.json ← generation eval dataset (44 answerable)
│   ├── results/              ← frozen JSONL per configuration + judged outputs
│   ├── metrics/              ← computed metrics JSON + comparison/citation reports
│   ├── experiment_notes.md   ← experiment log (config → result → decision)
│   ├── failures.md / generation_failures.md / citations_failures.md
│
└── data/
    ├── pdfs/Project_pdf.pdf  ← source corpus (NICE NG217, 161 pages)
    └── chroma_db/            ← baseline store + experiments/ (isolated stores)
```

Key files to read first: `PLAN.md` (authoritative plan), `evaluation/metrics/comparison.md`
(experiment history + final configuration justification), `evaluation/experiment_notes.md`
(hypothesis → decision log).

---

## Reproducing the evaluation

```text
run retrieval → (LLM judge) label → compute metrics → rank analysis
```

### Retrieval (canonical pipeline, two steps)

```bash
# Step 1: candidate pool (dense Top-20 + BM25 Top-20)
python scripts/run_hybrid_evaluation.py \
    --store data/chroma_db/experiments/large_800_100 \
    --name hybrid_800_100

# Step 2: cross-encoder reranking → final Top-10
python scripts/run_rerank_evaluation.py \
    --candidates evaluation/results/hybrid_800_100_candidates.jsonl \
    --name reranked_hybrid_800_100
```

### Label and compute metrics

```bash
python scripts/label_results.py evaluation/results/reranked_hybrid_800_100_top10.jsonl
python scripts/metrics.py evaluation/results/reranked_hybrid_800_100_top10_labeled.jsonl
python scripts/analyze_topk.py evaluation/results/reranked_hybrid_800_100_top10_labeled.jsonl
```

All scripts read only frozen artifacts and write derivative files — nothing frozen is
modified. Labeling needs `.env` keys; retrieval/metrics/analysis run offline.

### Generation and citations (read-only reuse of frozen outputs)

The frozen Generation V2 outputs are `evaluation/results/generation_v2.jsonl` (answers)
and `generation_v2_judged.jsonl` (judged). They can be regenerated with the existing
scripts (v2 uses the current `GROUNDING_SYSTEM_PROMPT` in `src/generation.py`; the v1→v2
change is documented in `evaluation/metrics/generation_prompt_comparison.md`):

```bash
python scripts/run_generation_evaluation.py \
    --output evaluation/results/generation_v2.jsonl
python scripts/evaluate_generation.py \
    --input evaluation/results/generation_v2.jsonl \
    --output evaluation/results/generation_v2_judged.jsonl

python scripts/run_citations.py \
    --input evaluation/results/generation_v2.jsonl \
    --output evaluation/results/citations_v1.jsonl
python scripts/evaluate_citations.py evaluation/results/citations_v1.jsonl
```

`evaluate_citations.py` is resumable and falls back to a deterministic token-overlap judge
when the LLM API is unavailable (verdicts tagged `judge=deterministic` and reported
separately).

---

## Evaluation methodology

- **LLM-as-a-Judge relevance labeling:** each retrieved `(question, chunk)` pair is sent to
  an LLM judge (`deepseek/deepseek-v4-flash`, temperature 0) which returns `relevant` /
  `not_relevant` plus a one-sentence reason (rubric in `src/judge.py`).
- The judge measures **retrieval relevance** ("does this chunk contribute toward answering
  the question?") — **not** clinical correctness.
- These labels are a **proxy for relevance**, not clinically validated ground truth.

## Evaluation limitations

- Labels come from an **LLM judge**, not clinical validation — engineering evidence, not
  clinical proof.
- Small evaluation sets: 20 benchmark + 27 holdout questions; Wilson 95% CIs are wide
  (e.g. benchmark P@3 CI ≈ [0.39, 0.64]).
- Some **judge-label variance** (~5–7% label flips on repeated runs) sets a noise floor for
  small metric differences.
- **Q20** (benchmark), **H26**, and **H27** (holdout) are **deliberate negative cases**
  (dosing / drug cost not covered by the guideline). Do not read them as ordinary failures.
- P@K / Hit@K / candidate recall are **retrieval** metrics, not answer accuracy.
- The deterministic citation support judge measures lexical overlap, not semantic
  entailment — only the LLM judge measures semantic support.

---

## Metrics explained (beginner-friendly)

For a set of `N` questions, each with a ranked Top-K of chunks:

- **Precision@K (P@K)** — of the top `K` chunks, what fraction are relevant?
  `P@3 = 0.517` means, on average, ~1.55 of the top-3 chunks are relevant.
- **Hit@K** — on what fraction of questions does *at least one* relevant chunk appear in
  the top `K`? `Hit@5 = 0.95` means 19 of 20 questions have relevant evidence in their
  top-5.
- **MRR (Mean Reciprocal Rank)** — how high is the *first* relevant chunk on average?
  `MRR = 0.879` means the first relevant chunk typically appears near the top.
- **Candidate recall** — of the questions that have relevant evidence in the guideline, on
  what fraction does that evidence exist anywhere in the candidate union (dense Top-20 ∪
  BM25 Top-20)? `19/19` benchmark and `25/25` holdout mean recall is solved: the evidence
  is always in the pool; the remaining work is ranking it to the top.

---

## Dependencies

See `requirements.txt` (pinned). Highlights: `python-dotenv`, `pypdf`, `langchain` family,
`sentence-transformers`, `langchain-chroma`, `chromadb`, `langchain-openai`, `rank_bm25`,
`fastapi`, `uvicorn`, and `gradio` (UI only, not the production backend). Models are
downloaded from Hugging Face on first use.

---

*Retrieval metrics are LLM-judged evidence, not clinical validation. Safety guardrails are
in progress and not yet implemented. See `PLAN.md` for the full plan and phase status.*