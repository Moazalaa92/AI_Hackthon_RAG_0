# Hackathon Lectures — RAG over a NICE Epilepsy Guideline (NG217)

A LangChain-based **retrieval-augmented generation (RAG)** project that answers questions
from `data/pdfs/Project_pdf.pdf` (NICE NG217, "Epilepsies in children, young people and
adults", 161 pages).

**Current status: the retrieval half of the RAG system is built and validated.**
Retrieval finds the right evidence chunks in the guideline for a question. The Generation
half (turning those chunks into a natural-language answer) is **not implemented yet** — that
is the next phase.

---

## Project status

| Component | Status |
|---|---|
| **Retrieval** | **VALIDATED / FROZEN FOR NOW** |
| **Generation** | NOT STARTED (Phase 8 — next) |
| **Sources / citations** | NOT STARTED (Phase 9) |
| **End-to-end pipeline** | NOT STARTED (Phase 10) |
| **UI** | NOT STARTED (Phase 11) |

Phases 1–7 (concepts → setup → ingestion → chunking → embeddings → vector store →
retrieval) are DONE. Phase 12 (retrieval evaluation) and Phase 13 (retrieval optimization)
are DONE for the current iteration. See `PLAN.md` for the authoritative phase-status table.

This is **not** yet a complete RAG QA application. Today it answers the question
*"which parts of the guideline are relevant to this question?"* — not *"here is the answer."*

---

## Validated retrieval architecture (frozen)

```text
PDF
 ↓
Ingestion
 ↓
Chunking: 800 / 100 (characters, overlap)
 ↓
Embedding: sentence-transformers/all-MiniLM-L6-v2
 ↓
 ┌─────────────────────────────┐
 │                             │
 ↓                             ↓
Dense Top-20              BM25 Top-20
(Chroma, L2)              (lexical)
 │                             │
 └──────────────┬──────────────┘
                ↓
         Candidate Union
                ↓
   Cross-Encoder Reranker
   cross-encoder/ms-marco-MiniLM-L-6-v2
                ↓
           Final Top-10
```

**Important — RRF is NOT part of the final architecture.**

The historical hybrid experiment fused dense + BM25 with **Reciprocal Rank Fusion (RRF,
k=60)**, but that was an experiment, not the chosen design. RRF capped single-retriever
chunks at ~1/(k+1) and hurt recall of BM25-only evidence (the Q09 case). The final
architecture replaces RRF with a learned **cross-encoder** that scores every
`(question, chunk)` pair directly. The historical RRF artifacts are preserved for
reproducibility; do not reintroduce RRF into the final pipeline.

### Configuration summary

| Setting | Value |
|---|---|
| Chunk size / overlap | `800` / `100` characters |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Dense retrieval | Chroma, **L2 distance**, Top-20 |
| BM25 | lexical, built from the **same** stored chunks, Top-20 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Final retrieval | Top-10 |

Score semantics:

- **Chroma L2** — lower distance = more similar (rank 1 = smallest distance).
- **Cross-encoder** — higher score = more relevant (the final ranking signal).

---

## Validated results

Relevance labels come from an **LLM-as-a-Judge** (see [Evaluation methodology](#evaluation-methodology));
these are retrieval-quality numbers, **not answer accuracy**.

### Original 20-question benchmark (`evaluation/dataset.json`)

```text
P@3    = 0.5167
P@5    = 0.4000
Hit@3  = 0.90
Hit@5  = 0.95
Hit@10 = 0.95
```

### Independent 27-question holdout (`evaluation/holdout_dataset_v1.json`)

```text
P@3    = 0.5802
P@5    = 0.4519
Hit@3  = 0.9259
Hit@5  = 0.9259
Hit@10 = 0.9259
```

### Answerable candidate recall (relevant evidence found in the Top-20+Top-20 union)

```text
Benchmark: 19/19 answerable questions = 100%
Holdout:   25/25 answerable questions = 100%
```

> The higher holdout P@3 is **not** a system improvement. It is the *same frozen
> architecture* evaluated on a different dataset. The holdout exists to confirm the
> architecture generalizes; it was not used to tune anything.

Full details: `evaluation/metrics/comparison.md` (experiment history + conclusions),
`evaluation/experiment_notes.md` (hypothesis → decision log), `evaluation/failures.md`
(failure analysis).

---

## Repository structure

```text
hackathon_lectures/
├── README.md                 ← this file
├── PLAN.md                   ← full project plan: phase status, architecture, source inventory
├── requirements.txt          ← pinned Python dependencies
├── .env.example              ← template for local configuration (copy to .env)
├── .gitignore                ← ignores .env, .venv, __pycache__, generated stores
├── app.py                    ← legacy demo (ingest + dense retrieve, no LLM) — NOT canonical
│
├── src/                      ← core modules
│   ├── config.py             ← reads .env, path/model defaults
│   ├── ingestion.py          ← PDF → pages (PyPDFLoader)
│   ├── chunking.py           ← pages → chunks (RecursiveCharacterTextSplitter)
│   ├── embeddings.py         ← all-MiniLM-L6-v2 wrapper
│   ├── vectorstore.py        ← Chroma persistence + query
│   ├── retrieval.py          ← dense Top-K retrieval (baseline path)
│   ├── hybrid_retrieval.py   ← dense + BM25 candidate union (historical experiment)
│   ├── reranking.py          ← cross-encoder reranker (final stage)
│   ├── evaluation.py         ← run fixed dataset retrieval → JSONL
│   ├── metrics.py            ← Precision@K, Hit@K, Top-K analysis
│   └── judge.py              ← LLM relevance judge (LLM-as-a-Judge)
│
├── scripts/                  ← command-line tools
│   ├── ingest.py             ← build a Chroma store from the PDF
│   ├── retrieve.py           ← demo retrieval (5 built-in questions, dense-only)
│   ├── run_evaluation.py     ← dense-only Top-K on the eval dataset
│   ├── run_hybrid_evaluation.py    ← dense+BM25 candidate pool (historical)
│   ├── run_rerank_evaluation.py    ← cross-encoder rerank → final Top-10
│   ├── label_results.py      ← LLM-judge labels for a results JSONL
│   ├── metrics.py            ← P@3/P@5 from a labeled file
│   ├── analyze_topk.py       ← Top-3/5/10 rank analysis from a labeled file
│   ├── candidate_recall_analysis.py ← recall of the union candidate pool
│   ├── rerank_analysis.py    ← rank movement: fusion → reranked
│   └── inspect_*.py          ← inspect documents / chunks / vectors / results
│
├── evaluation/               ← frozen evaluation artifacts
│   ├── dataset.json          ← canonical 20-question benchmark
│   ├── holdout_dataset_v1.json ← independent 27-question holdout
│   ├── results/              ← frozen retrieval JSONL per config + labels
│   ├── metrics/              ← computed metrics JSON + comparison.md
│   ├── experiment_notes.md   ← experiment log (config → result → decision)
│   └── failures.md           ← retrieval failure analysis
│
└── data/
    ├── pdfs/Project_pdf.pdf  ← source corpus (NICE NG217, 161 pages)
    └── chroma_db/            ← baseline store + experiments/ (isolated per-config stores)
```

Key files to read first:

- `PLAN.md` — the authoritative plan (phase status, architecture, deviations, metric
  definitions, source/script inventory).
- `evaluation/experiment_notes.md` — what was tried, why, and what was decided.
- `evaluation/metrics/comparison.md` — the full experiment comparison and the final
  configuration justification.
- `evaluation/failures.md` — why specific retrieval failures happened and how they were
  addressed.
- `src/judge.py` — the LLM judge rubric (defines "relevant" vs "not_relevant").

---

## Quick Start (clone → install → configure → ingest → retrieve)

### 1. Clone the repository

```bash
git clone <repository-url>
cd hackathon_lectures
```

### 2. Create and activate a virtual environment

Requires **Python 3.10+** (developed on 3.10).

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs LangChain (community, text-splitters, huggingface, chroma, openai),
`sentence-transformers`, `chromadb`, `pypdf`, `python-dotenv`, `rank_bm25`, and `gradio`.
The first run also downloads the embedding and cross-encoder models from Hugging Face.

### 4. Configure the environment

```bash
cp .env.example .env
```

`.env` is local-only and **gitignored** — never commit it.

- **Retrieval, ingestion, and metrics need no API key.** All of that runs offline (the
  embedding and reranker models are local Hugging Face models).
- **LLM-based labeling and Generation need a working API key.** `scripts/label_results.py`
  calls an LLM judge through OpenRouter.

`.env.example` variables:

| Variable | Required for | Notes |
|---|---|---|
| `LLM_API_KEY` | Labeling / Generation | OpenRouter API key. **Do not commit.** |
| `LLM_BASE_URL` | Labeling / Generation | e.g. `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | Labeling / Generation | e.g. `deepseek/deepseek-v4-flash` |
| `DATA_DIR` | Optional | Defaults to `data` |
| `CHROMA_DB_DIR` | Optional | Defaults to `data/chroma_db` |
| `PDF_DIR` | Optional | Defaults to `data/pdfs` |

The configuration is read by `src/config.py`. All of the following have safe defaults and
can be left unset unless you use a custom layout: `DATA_DIR`, `CHROMA_DB_DIR`, `PDF_DIR`,
plus `TOP_K` (5), `EVAL_TOP_K` (10), `EVALUATION_DIR`, `RESULTS_DIR`, `METRICS_DIR`,
`EXPERIMENTS_DIR`.

### 5. Build the vector store (ingest the PDF)

Retrieval needs the PDF embedded into Chroma first. The workflow is:

```text
PDF → load pages → chunk (800/100) → embed (MiniLM) → store in Chroma
```

To reproduce the **current architecture's store** (800/100, isolated from the baseline):

```bash
python scripts/ingest.py --chunk-size 800 --chunk-overlap 100 \
    --persist-dir data/chroma_db/experiments/large_800_100
```

- Defaults (`python scripts/ingest.py` with no arguments) build the **baseline** store
  (500/50) at `data/chroma_db/`.
- Experiment stores live in `data/chroma_db/experiments/<config>/` so they never disturb
  the baseline. The `experiments/` directory is gitignored (rebuildable).
- Use `--embedding` to pick a different embedding model; it must match the model used when
  querying the same store.

### 6. Run a retrieval query

The demo CLI is `scripts/retrieve.py`:

```bash
python scripts/retrieve.py
```

It runs **five built-in example questions** and prints the dense Top-K chunks (rank, L2
score, chunk id, source page, excerpt) for each. Note:

- It does **not** take a question as a command-line argument (there is no `"question"`
  argument); the questions are hardcoded in the script.
- It uses the **baseline** store by default and dense retrieval only (`TOP_K=5`).

To inspect a **specific** question against a **specific** store from Python:

```python
from src.retrieval import retrieve

for doc, score in retrieve(
    "Which medicine is offered as first-line treatment for absence seizures?",
    top_k=10,
    persist_dir="data/chroma_db/experiments/large_800_100",
):
    print(score, doc.metadata["chunk_id"], doc.page_content[:120])
```

What to expect:

```text
question → retrieve candidates → rank chunks → inspect Top-K chunks
```

This retrieves **evidence only**. It does not generate a natural-language answer —
Generation is not implemented yet.

---

## Reproducing the evaluation

The pipeline is:

```text
run retrieval → (LLM judge) label → compute metrics → rank analysis
```

### 1. Run retrieval for the benchmark dataset

Dense-only Top-K against a store (no LLM):

```bash
python scripts/run_evaluation.py \
    --store data/chroma_db/experiments/large_800_100 \
    --name rerun_800_100
```

Writes `evaluation/results/rerun_800_100_top10.jsonl` (200 records: 20 questions × top-10).

### 2. Reproduce the full frozen pipeline (candidate union → cross-encoder)

Two steps — first build the candidate pool (dense Top-20 + BM25 Top-20), then rerank:

```bash
# Step 1: candidate generation (dense + BM25 union). No reranker here.
python scripts/run_hybrid_evaluation.py \
    --store data/chroma_db/experiments/large_800_100 \
    --name hybrid_800_100

# Step 2: cross-encoder reranking over the frozen candidate pool → final Top-10
python scripts/run_rerank_evaluation.py \
    --candidates evaluation/results/hybrid_800_100_candidates.jsonl \
    --name reranked_hybrid_800_100
```

The first step writes the candidate union (`..._candidates.jsonl`) and an RRF-ranked Top-10
(historical step; this file is *not* the final ranking). The second step writes the **final
Top-10** (`..._top10.jsonl`) ranked purely by the cross-encoder score.

> RRF appears only in step 1 as the historical fusion inside `run_hybrid_evaluation.py`.
> The final architecture's ranking is the cross-encoder (step 2). See the architecture
> section above.

### 3. Label results with the LLM judge (requires `.env` keys)

```bash
python scripts/label_results.py evaluation/results/reranked_hybrid_800_100_top10.jsonl
```

- Reads only the frozen retrieval JSONL (never modifies it).
- Writes a **derivative** `<name>_labeled.jsonl` next to the input.
- Resumable/idempotent: already-labeled records are skipped on re-run.

### 4. Compute metrics

```bash
python scripts/metrics.py evaluation/results/reranked_hybrid_800_100_top10_labeled.jsonl
python scripts/analyze_topk.py evaluation/results/reranked_hybrid_800_100_top10_labeled.jsonl
```

- `metrics.py` prints per-question and mean **P@3 / P@5** and writes
  `evaluation/metrics/<experiment>_p3.json` and `..._p5.json`.
- `analyze_topk.py` prints the Top-3/5/10 **Hit@K** analysis and writes
  `evaluation/metrics/<experiment>_topk.json`.

### 5. (Optional) Analysis of the historical experiments

```bash
# Candidate recall: did the union find relevant chunks that dense alone missed?
python scripts/candidate_recall_analysis.py \
    evaluation/results/hybrid_800_100_candidates_labeled.jsonl

# Rank movement: fusion rank vs cross-encoder rank for pool-relevant chunks
python scripts/rerank_analysis.py
```

Both read only frozen artifacts and write to `evaluation/metrics/`.

---

## Frozen evaluation artifacts

| File | What it is |
|---|---|
| `evaluation/dataset.json` | Canonical **20-question benchmark** (Q01–Q20), used throughout the experiments. |
| `evaluation/holdout_dataset_v1.json` | Independent **27-question holdout** (H01–H27), created after the architecture was chosen. |
| `evaluation/results/*.jsonl` | Frozen retrieval output per configuration (unlabeled). |
| `evaluation/results/*_labeled.jsonl` | Derivative files with LLM-judge labels. |
| `evaluation/metrics/*.json` | Computed metrics (P@3, P@5, Top-K) per configuration. |
| `evaluation/metrics/comparison.md` | Experiment comparison + final configuration justification. |
| `evaluation/experiment_notes.md` | Hypothesis → decision log for every experiment. |
| `evaluation/failures.md` | Retrieval failure analysis. |

Why two datasets exist:

- The **benchmark** was used during the experiments to pick the architecture.
- The **holdout** was used only *afterward* to test whether the chosen architecture
  generalizes. It was **not** used to tune anything, so it is an honest generalization test.

---

## Evaluation methodology

- **LLM-as-a-Judge relevance labeling:** each retrieved `(question, chunk)` pair is sent to
  an LLM judge (`deepseek/deepseek-v4-flash`, temperature 0, up to one retry) which returns
  `relevant` or `not_relevant` plus a one-sentence reason. Rubric in `src/judge.py`.
- The judge evaluates **retrieval relevance** — "does this chunk contribute toward
  answering the specific question?" — not clinical correctness.
- These labels are a **proxy for relevance**, not clinically validated ground truth.
- Labeling requires an OpenRouter key in `.env`; retrieval, metrics, and analysis do not.

---

## Metrics explained (beginner-friendly)

For a set of `N` questions, each with a ranked Top-K of chunks:

- **Precision@K (P@K)** — of the top `K` chunks, what fraction are relevant?
  `P@3 = 0.5167` means, on average, ~1.55 of the top-3 chunks are relevant.
- **Hit@K** — on what fraction of questions does *at least one* relevant chunk appear in
  the top `K`? `Hit@3 = 0.90` means 18 of 20 questions have relevant evidence in their
  top-3.
- **Candidate recall** — of the questions that *have* relevant evidence in the guideline,
  on what fraction does that evidence exist anywhere in the candidate union (dense Top-20 ∪
  BM25 Top-20)? `19/19` benchmark and `25/25` holdout means recall is solved: the
  evidence is always in the pool; the remaining work is ranking it to the top.

**P@K and Hit@K are retrieval metrics. They are NOT answer accuracy.** They measure whether
the right evidence chunks surface, not whether a correct final answer would be produced.

---

## Evaluation limitations

- Relevance labels come from an **LLM judge**, not clinically validated ground truth —
  engineering evidence, not clinical validation.
- Small evaluation sets: **20** benchmark questions and **27** holdout questions; Wilson
  95% CIs are wide (e.g. benchmark P@3 CI ≈ [0.39, 0.64]).
- Some **judge-label variance** was observed across repeated runs (~5–7% label flips),
  setting a noise floor for small metric differences.
- **Q20** (benchmark), **H26**, and **H27** (holdout) are **deliberate negative cases**
  (dosing / drug cost not covered by the guideline). Do not read them as ordinary
  retrieval failures.
- P@3 / Hit@K / candidate recall are **retrieval** metrics, not answer accuracy.
- One real ranking weakness remains: occasional top-3 precision density drops (a distractor
  outranking evidence), e.g. the Q11 lexical-overlap case. It did not reproduce on the
  holdout.

---

## Canonical vs historical components

| Component | Role |
|---|---|
| `src/retrieval.py` | Dense-only baseline path (canonical for simple dense queries). |
| `src/hybrid_retrieval.py` | **Historical experiment** — dense + BM25 + RRF fusion. Kept for reproducibility. |
| `src/reranking.py` | **Canonical final stage** — cross-encoder over the candidate union. |
| `src/evaluation.py`, `src/metrics.py`, `src/judge.py` | Canonical evaluation pipeline. |
| `baseline`, `large_800_100`, `large_500_75`, `mpnet`, `hybrid+RRF`, `reranked` | **Historical experiments**, with their stores, results, and metrics preserved. |

The experiment artifacts are intentionally kept in the repo so the results are auditable
and the reasoning (hypotheses → decisions) is traceable in `experiment_notes.md`.

---

## What is NOT implemented yet

- **Generation** (Phase 8) — turning retrieved chunks into a natural-language answer.
- **Sources / citations** (Phase 9) — surfacing which chunk(s) support the answer.
- **End-to-end pipeline** (Phase 10) — question → answer in one command.
- **UI** (Phase 11).

The next phase is **Phase 8 — Generation**. Do not change the frozen retrieval
architecture (chunking 800/100, MiniLM, dense/BM25 depth 20, cross-encoder, top-10 depth)
before starting it.

---

## Dependencies

See `requirements.txt` (pinned). Highlights: `python-dotenv`, `pypdf`, `langchain`,
`langchain-community`, `langchain-text-splitters`, `langchain-huggingface`,
`sentence-transformers`, `langchain-chroma`, `chromadb`, `langchain-openai`, `gradio`,
`rank_bm25` (added for the historical hybrid experiment). Models are downloaded from
Hugging Face on first use.

---

*Retrieval metrics are LLM-judged evidence, not clinical validation. See `PLAN.md` for the
full plan and phase status.*
