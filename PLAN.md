# RAG Master Plan — Build a PDF Question-Answering System with LangChain

A progressive, teach-as-we-build plan. We implement **one phase at a time**, verify each one,
and never move on until the current phase is confirmed working.

> This document is the reference for the whole project. Every phase ends with a
> "Definition of done". We do not start a phase until the previous one is verified.

---

## 0. The current state of the project

Before we plan, here is what already exists in this folder:

| File / folder      | What it is                                                      | Plan status                       |
| ------------------ | --------------------------------------------------------------- | --------------------------------- |
| `.env`             | LLM API key + base URL + model (OpenRouter / deepseek)          | Keep — we adapt the plan to it    |
| `requirements.txt` | `python-dotenv`, `pypdf`, `langchain`, `langchain-community`, `langchain-text-splitters`, `langchain-huggingface`, `sentence-transformers`, `langchain-chroma`, `chromadb`, `langchain-openai`, `gradio` | Phase 3–7 deps already installed |
| `data/pdfs/`       | `Project_pdf.pdf` — a NICE epilepsy guideline (NG217, 161 pages) | Use as the baseline test PDF      |
| `data/chroma_db/`  | Persisted Chroma store, collection `documents`, **L2 distance space** | Baseline store (rebuildable)      |
| `.venv/`           | Virtual environment                                             | Keep using it                     |
| `.gitignore`       | Ignores `.venv`, `.env`, `__pycache__`, `data/extracted/`       | Extend later for `data/chroma_db/experiments/` (Phase 13.3) |
| `src/`             | `config.py`, `ingestion.py`, `chunking.py`, `embeddings.py`, `vectorstore.py`, `retrieval.py` | Phases 3–7, all implemented      |
| `scripts/`         | `ingest.py`, `retrieve.py`, `inspect_docs.py`, `inspect_chunks.py`, `inspect_vectors.py`, `embed_demo.py` | Phase verification scripts       |
| `app.py`           | Naive end-to-end demo (PDF → store → retrieve)                  | NOT canonical; superseded by `src/` pipeline later |

**Current implementation status — Phases 2–7 are DONE:**

- Phase 2 Setup, Phase 3 Ingestion, Phase 4 Chunking, Phase 5 Embeddings, Phase 6 Vector
  store, Phase 7 Retrieval are **implemented and working** (marked `DONE` in section 9).
- `src/chunking.py` already accepts `chunk_size` / `chunk_overlap` parameters
  (`chunk_documents(documents, chunk_size=500, chunk_overlap=50)`) — ready for experiments.
- Metadata on every chunk: `chunk_id`, `source`, `page` (0-based), `page_label` (1-based),
  plus PDF-level fields (`title`, `author`, ...). **There is no `section` field** — the plan
  handles this explicitly in Phase 12.4.
- Chroma scores are **L2 distances**: lower = more similar; rank 1 = smallest distance.
- No evaluation code exists yet. The next work is the **Retrieval Evaluation / Optimization
  lab (Phases 12–13)**, which needs **no LLM** and can start immediately after Phase 7.

**A design decision made for us by the environment:** the original brief suggested Gemini,
but you already have an **OpenRouter** key configured. OpenRouter exposes an
OpenAI-compatible API, which means:
- We use the **same `langchain-openai` `ChatOpenAI` client** as if we were talking to OpenAI.
- We only override the `base_url` to point at OpenRouter.
- Trade-off: one API key gives us access to many models (deepseek, gpt, claude, etc.) so we
  can swap models easily later. Nothing else changes. Gemini would have meant a different
  client (`langchain-google-genai`); not necessary now.

---

## 1. What RAG is — in plain language

**RAG = Retrieval-Augmented Generation.**

A normal LLM answers from what it "memorized" while training on public data. It has **never
seen your PDF**. If you ask it about your PDF it will guess, and it may confidently invent
things (this is called **hallucination**).

RAG solves this by giving the LLM *your* data at answer time:

```
User question
   +  the most relevant passages from your PDFs
   →  LLM answers using ONLY that context
```

So instead of "ask the model from memory", we do:

```
Ask the model, and hand it the relevant pages of your documents to read first.
```

### Why RAG exists (three reasons)

1. **Private data** — the model never sees your documents; they never go into its weights.
2. **Fresh data** — you can re-index your PDFs without retraining anything.
3. **Traceable data** — because the answer is built from retrieved chunks, we can show *where*
   the answer came from (sources/citations).

### RAG vs normal LLM prompting

| | Normal prompting | RAG |
|---|---|---|
| Input | question | question + retrieved chunks |
| Source of knowledge | model's memory | your documents |
| Hallucination risk | high | reduced (must be grounded in context) |
| Can cite sources | no | yes |
| Knows your PDFs | no | yes |

### The two big stages — commit these to memory

**Ingestion (offline, done once):** PDF → text → chunks → embeddings → vector store.

**Query (online, per question):** question → embedding → vector search → top-K chunks →
context → LLM → answer + sources.

Almost every "improvement" you will ever make to a RAG system changes one of these two
stages. Knowing *which* stage you are changing is the most important debugging skill.

---

## 2. The complete pipeline (visual)

### Conceptual — what actually happens

```
                    ┌──────────────┐
                    │     PDFs     │
                    └──────┬───────┘
                           ↓
                   text extraction          <- read bytes, get strings
                           ↓
                    text cleaning           <- fix whitespace, encoding, page breaks
                           ↓
                       chunking             <- cut text into retrievable pieces
                           ↓
                      embeddings            <- turn each chunk into a vector
                           ↓
                     vector store           <- store vectors + text + metadata
                           │
                           │
User question              │
      ↓                    │
  query embedding          │
      ↓                    │
 semantic retrieval ←──────┘   <- find vectors most similar to the question
      ↓
  top-K chunks              <- the raw material of the answer
      ↓
  context construction      <- assemble chunks into a readable context block
      ↓
      prompt                <- instructions + context + question
      ↓
      LLM                   <- generates the answer, grounded in context
      ↓
  grounded answer
      ↓
  source attribution        <- map answer points back to file/page/chunk
      ↓
    Gradio UI               <- thin wrapper: upload PDFs, ask questions
```

### With LangChain — the same pipeline, named abstractions

```
PyPDFLoader
  → RecursiveCharacterTextSplitter
  → HuggingFaceEmbeddings (all-MiniLM-L6-v2)
  → Chroma (vector store)
  → retriever / similarity_search
  → context assembly (plain Python)
  → ChatPromptTemplate
  → ChatOpenAI (OpenRouter / deepseek)
  → answer + sources
```

Every LangChain class in the second diagram is *just a wrapper* around the raw steps in the
first diagram. Nothing magical. When we build each phase we will look at both sides.

---

## 3. Every component explained (definitions we will reuse)

| Term | Meaning in one sentence |
|---|---|
| **Document** | A LangChain object = `page_content` (text) + `metadata` (dict: source, page, ...). It is NOT a file; it is *one piece of text plus labels*. |
| **Chunk** | A small piece of text (e.g. 500 characters) that we can embed, retrieve, and show to the LLM. |
| **Embedding** | A list of numbers (a vector) that represents the *meaning* of a text. Similar meanings → similar vectors. |
| **Vector** | A fixed-length list of floats (our model: 384 numbers). |
| **Embedding model** | A neural network that converts text → vector. Must be the SAME model for documents and queries. |
| **Vector store** | A database that stores vectors and finds the nearest ones to a query vector. |
| **Similarity score** | How close two vectors are (embedding similarity is usually cosine; NOTE: Chroma's stored score is a distance — see below). |
| **Top-K** | "Give me the K most similar chunks" — e.g. Top-5 = the 5 best chunks. |
| **Retriever** | Whatever turns a question into a set of candidate chunks. Our first retriever = vector search. |
| **Context** | The retrieved chunks, joined into one text block given to the LLM. |
| **Prompt** | The instructions + context + question sent to the LLM. |
| **LLM** | The generative model that writes the answer. |
| **Grounding** | The answer is supported by the retrieved context (not invented). |
| **Hallucination** | The LLM states something not present in the context. |
| **Refusal** | The LLM correctly says "not found in the documents" instead of guessing. |
| **Metadata** | Labels attached to each chunk (filename, page, chunk id) that travel through the pipeline and end up as citations. |
| **Ingestion** | Offline phase: PDFs → vector store. |
| **Query** | Online phase: question → answer + sources. |

---

## 4. Architecture decisions (and why)

### 4.1 Ingestion and query are separated

Indexing is done once and persisted. Querying is fast and repeated. Keeping them as separate
scripts (`scripts/ingest.py`, `scripts/ask.py`) makes each easy to test alone.

### 4.2 The core pipeline is UI-independent

Gradio only *calls* `rag.query(question)`. It never contains RAG logic. This means you can
test everything from the terminal with `scripts/ask.py` and the UI is just a cosmetic layer.
If the UI breaks, the RAG system still works.

### 4.3 Retrieval is observable

We will write a script (`scripts/retrieve.py`) that shows **rank, score, chunk id, page,
filename, and text** for every question — with **zero LLM calls**. This is how you see what
the system *actually finds* before any model "thinks".

### 4.4 Retrieval quality and generation quality are judged separately

A good answer does not mean good retrieval. We always check retrieval first
("did the right chunk come back?") and generation second ("did the LLM use it correctly?").

### 4.5 Simple modules, not over-engineered packages

We will use **flat modules in one `src/` folder** rather than sub-packages like
`src/ingestion/`, `src/processing/` etc. A beginner project of this size does not need deep
folder nesting — it just makes navigation harder. We keep one file per concern.

### 4.6 Metadata must travel with every chunk

Without metadata (filename, page, chunk id) we cannot cite sources, cannot filter by
document, and cannot debug which PDF answered the question. Metadata is attached at chunking
time and must survive storage and retrieval.

---

## 5. Proposed project structure

```
rag-project/                      (this folder)
│
├── PLAN.md                       ← this master plan
├── README.md                     ← short usage notes (add later)
├── .env                          ← secrets (already exists, gitignored)
├── .env.example                  ← template with empty values (for other devs)
├── requirements.txt              ← dependencies (grows per phase)
│
├── data/
│   ├── pdfs/                     ← your PDFs (Project_pdf.pdf already here)
│   └── chroma_db/                ← persisted vector store (baseline, exists)
│       └── experiments/          ← Phase 13: one sub-folder PER chunk configuration
│           ├── baseline_500_50/
│           ├── large_800_100/
│           └── small_300_50/
│
├── src/
│   ├── __init__.py
│   ├── config.py                 ← reads .env, holds model/embedding settings (+ eval paths, Phase 12)
│   ├── ingestion.py              ← PDF → Documents (PyPDFLoader)
│   ├── chunking.py               ← Documents → Chunks (splitter + metadata) [DONE]
│   ├── embeddings.py             ← embedding model setup (all-MiniLM-L6-v2) [DONE]
│   ├── vectorstore.py            ← Chroma setup: add chunks, persist (+ persist_dir param, Phase 13)
│   ├── retrieval.py              ← question → top-K chunks (observable) [DONE]
│   ├── evaluation.py             ← Phase 12: run all questions, persist top-10 results (no LLM)
│   ├── metrics.py                ← Phase 12: Precision@K on labeled results
│   ├── generation.py             ← context + question → LLM answer (Phase 8, later)
│   ├── sources.py                ← citation resolution (marker → file/page/chunk) (Phase 9, later)
│   └── pipeline.py               ← rag.query(question) → Answer + Sources (Phase 10, later)
│
├── scripts/
│   ├── ingest.py                 ← CLI: index PDFs into Chroma (+ --chunk-size/--overlap/--persist-dir, Phase 13)
│   ├── inspect.py                ← CLI: inspect extracted pages / chunks
│   ├── retrieve.py               ← CLI: retrieval-only inspection (no LLM) [DONE]
│   ├── run_evaluation.py         ← Phase 12.2/12.3: run every eval question, persist top-10 results
│   ├── inspect_results.py        ← Phase 12.4: print full result records for a question
│   ├── label_results.py          ← Phase 12.5: interactive manual relevance labeling (persisted)
│   ├── metrics.py                ← Phase 12.6/12.7: Precision@3, Precision@5
│   ├── analyze_topk.py           ← Phase 12.8: Top-3 vs Top-5 vs Top-10 analysis
│   ├── run_experiment.py         ← Phase 13.2/13.3: build config store + run evaluation for it
│   ├── compare_configs.py        ← Phase 13.4: metric comparison table across configurations
│   ├── ask.py                    ← CLI: full RAG question → answer + sources (Phase 10, later)
│   └── evaluate.py               ← CLI: run the evaluation dataset, print metrics (superseded by run_evaluation.py)
│
├── evaluation/
│   ├── dataset.json              ← Phase 12.1: 20 fixed questions (reused by EVERY experiment)
│   ├── results/                  ← Phase 12.3: persisted top-10 retrievals (one JSONL per configuration)
│   ├── metrics/                  ← Phase 12.6/12.7 & 13.4: P@3/P@5 outputs + comparison table
│   ├── failures.md               ← Phase 12.9: documented retrieval failure analysis
│   └── experiment_notes.md       ← Phase 13: hypotheses, per-experiment configs, decisions
│
├── app/
│   └── gradio_app.py             ← thin UI calling pipeline.py (Phase 11, later)
│
├── experiments/
│   ├── final_configuration.md    ← Phase 13.6/13.7: chosen retrieval config + justification
│   └── phase1_notes.md           ← Phase 1 mental-model notes
│
└── tests/                        ← small sanity tests (pytest), optional but useful
```

**Why this shape (and what we dropped from the suggested structure):**
- Dropped `src/ingestion/`, `src/processing/`... sub-packages → single modules per concern.
  Simpler to read, fewer files, same boundaries.
- Dropped `tests/` until later — tests are valuable, but the *evaluation lab* (Phases 12–13)
  is the project's test harness: it measures retrieval objectively and is kept in `src/` and
  `scripts/` rather than a separate framework.
- Kept `experiments/` — this is where we record results of each improvement so we never lose
  track of "did the change help?"
- Every artifact that must be reused or inspected (dataset, results, labels, metrics) lives
  under `evaluation/`; every scratch/comparison note lives under `experiments/`.
- Experiment vector stores are isolated under `data/chroma_db/experiments/<config>/` so one
  chunk configuration can never accidentally query another configuration's index.

---

## 6. Dependencies — the minimum set, each explained

We add packages *only when a phase needs them*. Starting set (current file already has
`chromadb`, `gradio`, `openai`, `pypdf`, `python-dotenv`):

| Package | Purpose | Needed in phase |
|---|---|---|
| `python-dotenv` | Load `.env` into the environment | 2 (setup) |
| `pypdf` | Under-the-hood PDF text extraction used by the LangChain loader | 3 |
| `langchain` | Core orchestration classes and the `Runnable` pipeline | 2+ |
| `langchain-community` | Community integrations — contains `PyPDFLoader` (PDF → Documents) | 3 |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` (chunking) | 4 |
| `langchain-huggingface` | `HuggingFaceEmbeddings` wrapper around `sentence-transformers` | 5 |
| `sentence-transformers` | Downloads and runs `all-MiniLM-L6-v2` locally (no API needed) | 5 |
| `langchain-chroma` | LangChain's integration with the `chromadb` vector store | 6 |
| `chromadb` | The local vector database itself | 6 |
| `langchain-openai` | `ChatOpenAI` — talks to OpenRouter (OpenAI-compatible) | 8 |
| `openai` | Underlying SDK that `ChatOpenAI` uses (already installed) | 8 |
| `gradio` | The web UI | 11 |

**Notes:**
- `sentence-transformers` downloads `all-MiniLM-L6-v2` (~80 MB) from HuggingFace on first
  use. Needs internet once. Afterwards it runs fully offline — a good property.
- We do **not** need a dedicated vector-database server; Chroma persists to a local folder
  (`data/chroma_db/`).
- The `.env` already stores the OpenRouter key as `LLM_API_KEY`; `config.py` will read it.

---

## 7. LangChain ↔ underlying concept mapping

| Raw concept | LangChain class | What LangChain does for us | What it hides (we must know it) |
|---|---|---|---|
| Read PDF file | `PyPDFLoader` | Opens the file, uses `pypdf` to extract text per page | File I/O, PDF parsing quirks, page splitting |
| Split text into pieces | `RecursiveCharacterTextSplitter` | Splits on paragraphs→sentences→words, adds overlap | The chunk-size/overlap trade-off is OUR decision |
| Text → vector | `HuggingFaceEmbeddings` | Loads the model, batches text, returns vectors | What an embedding actually is; model must match |
| Store + search vectors | `Chroma` | Persists vectors + text + metadata; does the distance math | Indexing strategy, distance metric, K semantics |
| Question → chunks | `vectorstore.as_retriever()` | Thin wrapper around a similarity search | We will call `similarity_search_with_score` directly for transparency |
| Build prompt | `ChatPromptTemplate` | String templating | The actual text sent to the model is ours to write |
| Call the model | `ChatOpenAI` | HTTP call to OpenRouter, parses response | Tokens, context window, temperature, grounding prompt |
| Chain it together | `RunnableSequence` / plain functions | Pipes steps | Order of operations is the real logic |

**Design stance:** we use LangChain where it adds real value (loaders, splitter, embeddings,
vector store, LLM client) but we keep orchestration as plain Python functions so the data
flow stays visible. `rag.query(question)` is a normal function, not a magic chain.

---

## 8. How Gradio fits

Gradio is the **last** layer, added in Phase 11. It has two tabs:

1. **Ingest**: upload one or more PDFs → saved to `data/pdfs/` → indexed into the vector
   store. It calls the *same* `scripts/ingest.py` logic, not its own.
2. **Ask**: a text box → calls `rag.query(question)` → displays the answer, the retrieved
   chunks, and the citations (file + page + chunk id).

Rule: **Gradio contains zero RAG logic.** If we delete `app/gradio_app.py`, the entire system
still works from the terminal.

---

## 9. The 13 phases

Each phase block below includes: purpose, what we build, concepts you must know first,
technologies, verification, definition of done, and common mistakes. When we execute a phase
we will also follow the phase-completion format from section 13.

> **Reading order / implementation order note.**
> Phases 1–7 are **DONE** (Phase 1 conceptual, Phases 2–7 code). The original plan then
> continued with Generation/Sources/Pipeline/Gradio (Phases 8–11).
> Phases **12 (Retrieval Evaluation)** and **13 (Retrieval Optimization)** are a
> self-contained lab that depends **only on Phases 2–7** and requires **no LLM**. Per the
> lab brief, do **not** implement Generation just to reach evaluation. Implement Phases
> 12–13 immediately after Phase 7; Phases 8–11 (Generation, Sources, Pipeline, Gradio) can
> be built afterwards, using the same pipeline.

---

### Phase 1 — RAG architecture fundamentals (no code)

**Status: DONE — conceptual phase; the mental model is now captured by the implementation.**

**Purpose:** build the mental model before touching the keyboard.

**Concepts to learn:** what RAG is, why it exists, RAG vs normal prompting, ingestion vs
query, retrieval vs generation, hallucination vs grounding, where LangChain sits.

**Deliverable:** you can draw the pipeline from memory and explain what happens to the data at
every arrow.

**Verification:** you explain out loud (or write in `experiments/phase1_notes.md`) the answer
to: *"What happens when I ask the system a question?"* — from question text to the final
answer and sources.

**Definition of done:** you can explain the full data flow and the difference between a
retrieval failure and a generation failure.

---

### Phase 2 — Project setup

**Status: DONE — `src/config.py`, `.env.example`, directory skeleton, and
`requirements.txt` are in place and verified.**

**Purpose:** a clean, reproducible environment.

**Concepts to learn:** virtual environments, `requirements.txt`, `.env` vs `.env.example`,
configuration as a separate concern (`src/config.py`).

**Tasks:**
- Keep using `.venv`; extend `requirements.txt` with the LangChain packages (section 6).
- Create `.env.example` (empty values) and make sure `.env` stays gitignored.
- Create `src/config.py` that loads `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, and paths
  (`DATA_DIR`, `CHROMA_DIR`, `PDF_DIR`).
- Create the directory skeleton (section 5).

**Verification:** `python -c "import src.config; print(src.config.LLM_MODEL)"` prints the
model; `pip install -r requirements.txt` completes.

**Common mistakes:** committing `.env`; hard-coding paths; installing packages without
pinning versions.

**Definition of done:** every import works, config reads from `.env`, structure exists.

---

### Phase 3 — PDF ingestion

**Status: DONE — `src/ingestion.py` loads pages via `PyPDFLoader`; verified with
`scripts/inspect_docs.py` (161 pages from `Project_pdf.pdf`).**

**Purpose:** turn bytes on disk into structured text with page metadata.

**Concepts to learn:** LangChain `Document` (page_content + metadata), page numbers, why a
PDF page ≠ a good chunk, PDF extraction limitations (scanned PDFs have no text layer, tables
and columns come out jumbled).

**Build:** `src/ingestion.py` using `PyPDFLoader` → list of `Document`s (one per page), each
with `source` and `page` metadata.

**Under the hood:** `PyPDFLoader` calls `pypdf` to extract the text layer per page. If a PDF
is scanned (an image), there is **no text layer** and extraction returns empty/garble — that
is a real, common failure you will meet.

**Verification:** `scripts/inspect.py` prints, for the first few pages of
`Project_pdf.pdf`: page number, metadata, first 200 chars. Inspect page count and text
quality.

**Common mistakes:** assuming extraction is perfect; ignoring page metadata; forgetting that
OCR is a separate problem.

**Definition of done:** you can load `Project_pdf.pdf`, list its pages, and confirm real text
came out.

---

### Phase 4 — Text processing & chunking

**Status: DONE — `src/chunking.py` uses `RecursiveCharacterTextSplitter`
(500 / 50) and adds `chunk_id`; verified with `scripts/inspect_chunks.py`.
`chunk_documents(documents, chunk_size, chunk_overlap)` already accepts parameters
(needed for Phase 13 experiments).**

**Purpose:** split long text into retrievable, embeddable pieces.

**Concepts to learn:** chunk size, chunk overlap, recursive splitting (paragraph → sentence →
word), sentence/paragraph boundaries, why one whole PDF must NOT be one chunk (a 10-page
vector is "averaged meaning" — useless for retrieval), the size trade-off (small chunks =
precise but lose context; large chunks = contextual but noisy), incomplete sentences at chunk
edges, metadata preservation.

**Build:** `src/chunking.py` with `RecursiveCharacterTextSplitter`. Baseline starting point:
`chunk_size=500`, `chunk_overlap=50`. Keep `metadata` (add `chunk_id`).

**What happens inside:** the splitter walks the text trying separators in order
(`\n\n` → `\n` → `. ` → ` `) and cuts at ~500 characters, keeping ~50 characters of the
previous chunk at the start of the next so context survives the cut.

**Verification:** `scripts/inspect.py` shows each chunk: id, length, first/last 100 chars,
page. You will *see* overlap and (likely) a cut in the middle of a sentence — that is
expected, and it is exactly why overlap exists.

**Common mistakes:** chunk sizes that are too large/small; no overlap; destroying metadata;
not verifying actual chunk contents.

**Definition of done:** you can inspect real chunks and explain why the splitter cut where it
did.

---

### Phase 5 — Embeddings

**Status: DONE — `src/embeddings.py` wraps `all-MiniLM-L6-v2`
(384-dim, local); verified with `scripts/embed_demo.py`.**

**Purpose:** convert each chunk of text into a vector of numbers that encodes meaning.

**Concepts to learn:** what an embedding is (a point in a 384-dimensional space), how text
becomes a vector (the model encodes tokens → aggregates into one vector), why similar texts
get similar vectors, why dimensions are fixed (384 for `all-MiniLM-L6-v2`), **why documents
and queries MUST use the same model** (different models produce incomparable spaces), cosine
similarity vs distance, limitations (pure semantic matching misses exact keywords and
numbers).

**Build:** `src/embeddings.py` wrapping `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`.

**Why this model:** tiny (~80 MB), runs locally, strong enough for a baseline, embeddings are
comparable within it. Baseline only — we compare better models later, one at a time.

**Verification (the first real "aha"):** embed 3–4 short sentences, one pair near-synonyms
("How do I install pip?" vs "What is the procedure to set up pip?"), one unrelated, and print
their pairwise cosine similarity. Related pairs should score high (>0.7), unrelated low
(<0.3).

**Common mistakes:** mixing embedding models; not normalizing; expecting exact-match behavior
from a semantic model.

**Definition of done:** you can compute embeddings and similarities by hand and explain what
the numbers mean.

---

### Phase 6 — Vector store (Chroma)

**Status: DONE — `src/vectorstore.py` persists to `data/chroma_db/`
(collection `documents`), uses `chunk_id` as the document ID for idempotent
re-indexing; verified with `scripts/inspect_vectors.py`. Note: the store uses
Chroma's default **L2 distance space**, so scores returned are distances —
**lower = more similar**.**

**Purpose:** persist chunks + embeddings + metadata, then search them.

**Concepts to learn:** what a vector store stores (embedding vector, original text, metadata),
why we store the original text (the LLM needs words, not vectors; and the vector alone cannot
be shown to a human), why metadata, how similarity search works (embed query → compare against
every stored vector → return closest), what Top-K means, what the returned score means
(L2 distance in Chroma: lower = more similar).

**Build:** `src/vectorstore.py` with `Chroma(persist_directory="data/chroma_db/")`. Add chunks
with embeddings + metadata.

**Under the hood:** Chroma persists to SQLite + binary index files in the folder. It
computes a distance between the query vector and all stored vectors and ranks them. The
current store uses Chroma's **default L2 (Euclidean) distance space**, so the score from
`similarity_search_with_score` is a distance — **lower = more similar**.

**Verification:** insert `Project_pdf.pdf` chunks, then run a tiny test that searches for
several terms and prints results with scores. Confirm persistence by re-opening the store in
a fresh process and searching again.

**Common mistakes:** re-indexing duplicates (add without clearing); forgetting persistence
directory; confusing the score (Chroma returns *distance*).

**Definition of done:** you can add to and search Chroma directly, explain Top-K and scores,
and see data persist across runs.

---

### Phase 7 — Retrieval (the heart of the system)

**Status: DONE — `src/retrieval.py` uses `similarity_search_with_score`
(no LLM); `scripts/retrieve.py` prints rank, score, chunk id, page, and text.
Note: current `TOP_K` default is 5 (from `.env`); the evaluation lab uses
top-10 explicitly (Phase 12.2).**

**Purpose:** turn a question into ranked, inspected chunks — **with no LLM involved**.

**Concepts to learn:** query embedding, semantic search, Top-K, why rank matters, what a score
means, the difference between *retrieval failure* (right chunk ranked too low) and
*generation failure* (right chunk retrieved but the LLM answers wrong). This is where most
real RAG problems live.

**Build:** `src/retrieval.py` using `similarity_search_with_score` (NOT the default retriever
abstraction) so we can see the scores. `scripts/retrieve.py` prints a table:

```
Q: "How is project funding allocated?"
  #1  0.78  chunk_104  Project_pdf.pdf  p.12  "Funding is allocated..."
  #2  0.85  chunk_189  Project_pdf.pdf  p.18  ...
```

**Verification:** ask 5–10 questions, several of which you *know* are in the PDF, and inspect
whether the right chunks appear and at what rank.

**Common mistakes:** trusting a nice answer without checking retrieval; not printing scores;
using default `K` without reasoning about it.

**Definition of done:** you can explain, for a given question, exactly which chunks came back,
why they came back, and what the scores mean — with no LLM in the loop.

---

> **Lab note:** Phases 8–11 (Generation, Sources, Pipeline, Gradio) complete the original RAG
> pipeline. The retrieval evaluation lab (Phases 12–13) does **not** need them and must not
> wait for them. You may implement Phases 12–13 now and return to 8–11 afterwards. The next
> `Next Implementation Sequence` section (end of this document) gives the exact order.

---

### Phase 8 — Generation

**Purpose:** turn question + retrieved context into a grounded answer.

**Concepts to learn:** context construction (joining chunks), the prompt (system instructions,
context, question), context window limits, grounding, hallucination, and **refusal behavior**.
Prompt design that explicitly says: *"Answer only from the context. If the context does not
contain the answer, say 'not found in the documents'."*

**Build:** `src/generation.py`: assemble context from the retrieved chunks → build prompt with
`ChatPromptTemplate` → call `ChatOpenAI(base_url=OpenRouter, model=deepseek/v4-flash)`
→ return the answer. Keep chunk indices in the prompt (e.g. `[1]`, `[2]`) so the LLM can cite.

**Under the hood:** `ChatOpenAI` sends an HTTP request to OpenRouter with your messages and
streams back tokens. The context window (for this model, on the order of ~64k–128k tokens)
bounds how many chunks fit — that is why we only send top-K, not the whole document.

**Test with both cases:**
1. A question answerable from the PDF → answer should be grounded.
2. A question NOT in the PDF (e.g. "What is the chef's favorite pizza?") → the LLM must say
   it is not in the documents, not invent an answer.

**Common mistakes:** a weak prompt that lets the LLM ignore context; sending too much context
(overflow); not testing the refusal case; judging generation before verifying retrieval.

**Definition of done:** answerable questions get grounded answers; unanswerable ones are
correctly refused.

---

### Phase 9 — Source attribution (citations)

**Purpose:** map every part of the answer back to file + page + chunk.

**Concepts to learn:** why metadata must survive the whole pipeline, the citation scheme
(chunk `[n]` in the prompt → answer contains `[n]` → we resolve `[n]` to its filename, page,
and chunk id), and why citations let a user verify the answer.

**Build:** `src/sources.py`: resolve markers found in the answer into a source list, e.g.:

```
[1] → Project_pdf.pdf, page 12, chunk 104
```

Keep this simple: the prompt numbers the chunks, the LLM references `[1]`, `[2]`, and we
resolve the numbers against the retrieved list. Fancier citation frameworks are an
*advanced* step for later, not now.

**Verification:** for a few questions, confirm each citation points at the actual chunk the
answer came from, and the page number matches.

**Common mistakes:** metadata lost at chunking; markers not in the prompt; resolving against
the wrong retrieval.

**Definition of done:** every answer ships with citations that point to real, correct chunks.

---

### Phase 10 — RAG pipeline assembly

**Purpose:** combine components into one clean entry point:

```text
rag.query(question)  →  retrieval  →  context  →  prompt  →  LLM  →  citation resolution
                     →  Answer + Sources
```

**Build:** `src/pipeline.py` exposing `rag.query(question)` returning a small `Answer` object
`(answer, sources, retrieved_chunks)`. `scripts/ask.py` prints the answer, then the sources
and the retrieved chunks.

**Important:** keep every component individually importable — `retrieval`, `generation`, and
`sources` must still work on their own (this is what makes debugging possible).

**Verification:** `python scripts/ask.py "your question"` returns answer + citations end to
end. Also verify components independently.

**Common mistakes:** fusing components into one big function; hiding scores from the caller;
coupling the pipeline to Gradio.

**Definition of done:** a single `rag.query(question)` call produces answer + sources, and
every stage is independently testable.

---

### Phase 11 — Gradio interface

**Purpose:** a thin UI over the working pipeline.

**Concepts to learn:** separating presentation from logic; upload flows; async-friendly
calls.

**Build:** `app/gradio_app.py`:
- **Upload tab** — save PDFs to `data/pdfs/`, call the ingestion logic.
- **Ask tab** — call `rag.query(question)`, display answer, sources, and retrieved chunks.

**Verification:** run `python app/gradio_app.py`, upload `Project_pdf.pdf`, ask a question,
see answer + citations.

**Common mistakes:** putting RAG logic in the UI; not re-indexing after uploading a new PDF;
blocking the UI thread on slow calls.

**Definition of done:** the UI works end to end and contains no RAG logic.

---

### Phase 12 — Retrieval Evaluation (the lab: measure before improving)

**Status: NOT STARTED — this is the retrieval evaluation lab. Requires only Phases 2–7,
no LLM, no UI.**

**Purpose:** objectively measure the retrieval system **before** any improvement. We build a
fixed, reusable evaluation dataset, run retrieval for every question, persist the top-10
results, manually label relevance, compute **Precision@3** and **Precision@5**, compare
Top-3/5/10, and document at least one genuine retrieval failure. The output is the baseline
that every future change is compared against. **No LLM is involved anywhere in this phase.**

**Concepts to learn:** evaluation datasets and stable question IDs; reproducibility (same
questions, same store, same query ⇒ same results); manual relevance labeling as ground
truth; the `Precision@K = (relevant chunks in top K) / K` formula; the tradeoff between
retrieving more context (Top-10) and introducing noise; retrieval failure vs generation
failure.

**Golden evaluation loop (applies to this phase and Phase 13):**

```text
Hypothesis → Experiment → Same evaluation questions → Retrieval
→ Manual relevance labels → Metrics → Analysis → Decision
```

Every claim must be backed by numbers, e.g. "Configuration A: P@3 = 0.73" — never
"this configuration looks better."

---

#### 12.1 Evaluation dataset

**Purpose:** a fixed set of ~20 questions, reused for every experiment so configurations are
compared fairly. Without a fixed dataset, results are not comparable and the "experiment"
is not reproducible.

**Concepts to learn:** curation (writing questions whose answers exist in the PDF),
question coverage/categories, stable IDs, JSON structure, keeping the dataset versioned.

**Implementation tasks:**
- Create `evaluation/dataset.json` with **20 questions** (15–20 acceptable) for
  `Project_pdf.pdf` (NICE NG217 epilepsy guideline, 161 pages).
- Cover different information types, at least: treatment recommendations; medication safety
  and warnings (e.g. valproate, topiramate); diagnosis and investigations (MRI, EEG);
  definitions/terminology; dosing/special populations; prognosis/risk; at least one question
  **not answerable** from the PDF (deliberate negative case).
- Each question gets a stable ID `Q01`–`Q20` and metadata describing what to look for.
- Example structure:

```json
{
  "dataset_version": 1,
  "created": "2026-08-17",
  "description": "Retrieval evaluation questions for Project_pdf.pdf (NICE NG217)",
  "questions": [
    {
      "id": "Q01",
      "question": "Which medicine is recommended as first-line treatment for Dravet syndrome?",
      "category": "treatment",
      "expected_pages": [12, 13],
      "notes": "Look for a recommendation about the first-line drug."
    }
  ]
}
```

**Expected files/modules:** new `evaluation/dataset.json` only. No `src/` changes.

**Data/artifacts:** `evaluation/dataset.json` is the single source of truth. It is read by
every evaluation run; it is never regenerated between experiments.

**Acceptance criteria:**
- 20 questions (minimum 15), IDs `Q01`…`Q20`, no duplicate IDs.
- Every question (except the deliberate negative case) is confirmed answerable from the PDF,
  with `expected_pages` matching real content.
- At least 6 distinct `category` values are represented.

**Verification:** print all IDs + questions (`python -c "import json; ..."`), then manually
spot-check several against the PDF pages.

**Dependencies:** Phase 7 (retrieval) so questions can be validated against real retrieval;
Phase 2 config paths.

---

#### 12.2 Evaluation runner (retrieval-only)

**Purpose:** run retrieval for **every** evaluation question with `top_k = 10` (at least 10),
using only the existing `retrieval`/`vectorstore` code. No LLM.

**Concepts to learn:** reproducible queries (fixed text, fixed store, fixed `k`), score
semantics (Chroma returns **L2 distance**, lower = more similar — rank 1 = smallest
distance), keeping the evaluation decoupled from the UI and from Generation.

**Implementation tasks:**
- Add evaluation settings to `src/config.py`:
  `EVAL_TOP_K` (default 10), `EVALUATION_DIR` (default `evaluation`), `RESULTS_DIR`,
  `LABELS_DIR` (if separate), `EXPERIMENTS_DIR` (default `data/chroma_db/experiments`).
- Create `src/evaluation.py` with two functions:
  - `run_retrieval(dataset, persist_dir, top_k=10) -> list[dict]` — for each question,
    call retrieval (reusing `src/retrieval.py`, parameterized by `persist_dir`), and build
    one result record per retrieved chunk (schema in 12.3).
  - `write_results(records, path)` — write the records as JSONL.
- Create `scripts/run_evaluation.py`: CLI that loads `evaluation/dataset.json`, runs
  retrieval against a given store, and writes results to `evaluation/results/`.

**Expected files/modules:** modify `src/config.py`; new `src/evaluation.py`,
`scripts/run_evaluation.py`. Reuse `src/retrieval.py`, `src/vectorstore.py`,
`src/embeddings.py` unchanged for the baseline.

**Data/artifacts:** `evaluation/results/baseline_500_50_top10.jsonl` (baseline run).

**Acceptance criteria:**
- Running the script produces **exactly** `20 questions × 10 chunks = 200` result records.
- The script is deterministic: re-running against the same store yields identical records.
- No LLM call is made (no `langchain-openai` import; no network call to OpenRouter).

**Verification:** run `python scripts/run_evaluation.py --store data/chroma_db`
and check the record count and a few records' `question_text` / `chunk_id` / `page`.

**Dependencies:** Phase 7; 12.1.

---

#### 12.3 Top-10 retrieval result persistence

**Purpose:** make every retrieved result reproducible and inspectable by persisting it, so
labeling, metrics, and failure analysis all operate on the *same* frozen results.

**Concepts to learn:** JSONL (one JSON object per line) vs JSON; why each field must be
stored (we must not re-run retrieval to "remember" what was retrieved); idempotent re-runs
(overwrite the file).

**Result record schema** (one per retrieved chunk):

```json
{
  "question_id": "Q01",
  "question_text": "...",
  "rank": 1,
  "score": 0.5957,
  "chunk_id": 290,
  "document_id": "chunk_290",
  "source": "data/pdfs/Project_pdf.pdf",
  "page": 72,
  "page_label": "73",
  "section": null,
  "chunk_text": "full text of the chunk...",
  "relevant": null
}
```

- `score` is the raw value from `similarity_search_with_score` (**L2 distance, lower =
  better**). Rank 1 always has the smallest score.
- `section` is `metadata.get("section")` — currently `null` (see 12.4).
- `relevant` starts as `null` and is set by manual labeling (12.5).

**Expected files/modules:** new `src/evaluation.py` (record builder + writer) already
created in 12.2; this sub-phase defines and freezes the schema. File naming convention:
`evaluation/results/<experiment_name>_top10.jsonl`.

**Data/artifacts:** `evaluation/results/baseline_500_50_top10.jsonl` with the full schema.

**Acceptance criteria:**
- Every record contains all required fields; `chunk_text` holds the full chunk text
  (not truncated); `section` is present (value or `null`); `relevant` is `null` initially.
- Re-running the evaluation overwrites the file cleanly (no duplicated lines).

**Verification:** `python -c` loads the JSONL, validates the schema for every line, and
prints field coverage (e.g. all records have `source` = the PDF path).

**Dependencies:** 12.2.

---

#### 12.4 Result inspection (display complete retrieval results)

**Purpose:** let a human read the *complete* retrieval result for any question: chunk text,
score, source, page, section, chunk ID, rank — reusing the existing metadata, no new
metadata system.

**Concepts to learn:** presentation vs logic; reading scores as distances; where the
`section` gap is and how we handle it.

**Implementation tasks:**
- Create `scripts/inspect_results.py`: given a results file and a question ID, print:

```text
Q01: Which medicine is recommended as first-line treatment for Dravet syndrome?
  #1  0.5957  chunk_290  data/pdfs/Project_pdf.pdf  p.73  section: N/A
      <full chunk text>
  #2  0.6644  chunk_334  data/pdfs/Project_pdf.pdf  p.82  section: N/A
      ...
```

- **Section handling (explicit):** the current metadata has **no `section` field**.
  Display `section: N/A` whenever `metadata.get("section")` is `None`. Do **not** invent a
  second metadata system. A future optional enhancement (heading-based section detection
  during chunking) is noted in Phase 13.5 but is not required for the lab.

**Expected files/modules:** new `scripts/inspect_results.py`. Reads only the persisted JSONL
— it must **not** re-run retrieval (inspection and labeling must use the same frozen data).

**Data/artifacts:** none new; consumes `evaluation/results/*.jsonl`.

**Acceptance criteria:**
- For any question in the dataset, the script prints all 10 records with: chunk text, score,
  source, page, section (or `N/A`), chunk ID, and rank.
- Output for a question matches `scripts/retrieve.py` for its top-5 rows (same ranks/scores).

**Verification:** compare `python scripts/inspect_results.py baseline_500_50_top10.jsonl Q01`
with `python scripts/retrieve.py` output for the same question (top-5 subset).

**Dependencies:** 12.3.

---

#### 12.5 Manual relevance labeling

**Purpose:** attach ground truth to every retrieved chunk so metrics can be computed, and
persist it so it can be reused by every metric run.

**Concepts to learn:** binary relevance (`relevant` / `not_relevant`); why labels must be
stored (not terminal output); why they must stay attached to the exact
question / chunk / rank; idempotent labeling.

**Implementation tasks:**
- Create `scripts/label_results.py`:
  - Walks a results file, and for each record with `"relevant": null` prints the question,
    rank, page, and full chunk text, then prompts `r` / `n` (relevant / not relevant).
  - Writes the choice **back into the record** in the JSONL (in place, with a `.bak` backup
    before the first edit).
  - Skips already-labeled records, so labeling is resumable and idempotent.
- **Label representation (exact):** string enum stored in the `relevant` field with exactly
  two values: `"relevant"` or `"not_relevant"` (`null` = not yet labeled).
  Definition: a chunk is `"relevant"` if it contains information that actually answers the
  question; partial/tangential chunks are `"not_relevant"`.

**Expected files/modules:** new `scripts/label_results.py`. Labels live **inside** the
persisted JSONL records — no separate label file, so a label can never detach from its
question/chunk/rank.

**Data/artifacts:** labeled `evaluation/results/*.jsonl` (the `relevant` column filled in).

**Acceptance criteria:**
- Every record in the baseline results file has `relevant` ∈ {`"relevant"`, `"not_relevant"`}
  (no `null` left).
- Labels survive restarts (they are read back from the file, not kept in memory).
- Re-running the labeling script changes nothing (idempotent).

**Verification:** run the script once for the baseline file; re-run it and confirm zero
prompts (all labeled); open the JSONL and confirm the `relevant` values.

**Dependencies:** 12.4 (inspection) so the human can judge chunks; 12.3 (persistence).

---

#### 12.6 Precision@3

**Purpose:** measure how many of the top-3 retrieved chunks are actually relevant, per
question and averaged.

**Concepts to learn:** the formula, why a fixed small K is a strict measure (top of the list
is what a user/LLM sees first).

**Formula (explicit):**

```text
Precision@3(question) = (number of "relevant" chunks among ranks 1..3) / 3
Average P@3 = (sum of per-question P@3) / (number of questions)
```

**Implementation tasks:**
- Create `src/metrics.py` with `precision_at_k(records, k=3)` returning per-question values
  and the average.
- Create `scripts/metrics.py` CLI that reads a labeled results file and prints:

```text
Per-question Precision@3:
  Q01  1.000
  Q02  0.667
  ...
Average Precision@3: 0.733
```

- Persist output to `evaluation/metrics/<experiment>_p3.json`.

**Expected files/modules:** new `src/metrics.py`, `scripts/metrics.py`.

**Data/artifacts:** `evaluation/metrics/baseline_500_50_p3.json` + printed table.

**Acceptance criteria:**
- Per-question P@3 and the average are computed from the **labeled** results only.
- Formula matches manual calculation for at least two questions.

**Verification:** hand-compute P@3 for `Q01` from its labels and compare with the script
output.

**Dependencies:** 12.5 (labels must exist).

---

#### 12.7 Precision@5

**Purpose:** same as 12.6 but over the top-5 — a slightly looser measure that tolerates one
bad chunk in the first five.

**Concepts to learn:** how P@K changes with K; why we report both P@3 and P@5.

**Formula (explicit):**

```text
Precision@5(question) = (number of "relevant" chunks among ranks 1..5) / 5
Average P@5 = (sum of per-question P@5) / (number of questions)
```

**Implementation tasks:** extend `src/metrics.py` (`k=5`), extend `scripts/metrics.py` to
print both tables (or a combined one). Persist `evaluation/metrics/<experiment>_p5.json`.

**Expected files/modules:** same as 12.6.

**Data/artifacts:** `evaluation/metrics/baseline_500_50_p5.json`.

**Acceptance criteria:** per-question and average P@5 computed from labeled results; matches
hand calculation for at least two questions.

**Verification:** same as 12.6.

**Dependencies:** 12.5.

---

#### 12.8 Top-3 vs Top-5 vs Top-10 analysis

**Purpose:** understand the tradeoff between retrieving additional useful context and
introducing irrelevant/noisy chunks as K grows. This is **separate** from P@3/P@5 (those are
aggregate precision numbers); this is a per-question *inspection* of what the extra ranks
bring in.

**Concepts to learn:** the "noise frontier"; why precision usually falls as K grows; reading
a precision curve; slicing persisted results (no re-retrieval needed).

**Implementation tasks:**
- Because top-10 results are already persisted and labeled, compute for **every** question
  (at least 3, preferably all 20 — no new code needed for more):
  - `P@3`, `P@5`, `P@10` per question.
  - Relevance of the *added* ranks: for ranks 4–5 (vs top-3) and ranks 6–10 (vs top-5),
    list whether they are `relevant` or `not_relevant`.
- Create `scripts/analyze_topk.py` printing, per question:

```text
Q07
  P@3 = 0.667  P@5 = 0.600  P@10 = 0.400
  ranks 4–5 relevance:  [not_relevant, relevant]
  ranks 6–10 relevance: [not_relevant, not_relevant, relevant, not_relevant, not_relevant]
```

**Expected files/modules:** new `scripts/analyze_topk.py` (reuses `src/metrics.py`).

**Data/artifacts:** `evaluation/metrics/baseline_500_50_topk.json` (optional) and a written
interpretation appended to `evaluation/experiment_notes.md`.

**Acceptance criteria:**
- Report covers at least 3 questions (design covers all 20 with no extra implementation).
- Each report shows P@3/P@5/P@10 and the relevance of the added ranks, plus one sentence on
  what the growth from 3→5→10 reveals (extra useful context vs noise).

**Verification:** spot-check one question by hand against its labeled JSONL records.

**Dependencies:** 12.5 (labels on all 10 ranks); 12.6/12.7 metrics.

---

#### 12.9 Retrieval failure analysis

**Purpose:** identify and document **at least one genuine retrieval failure** — the relevant
information was **not retrieved** or was **ranked too low**. This is the diagnostic core of
the lab and feeds the Phase 13 improvements.

**Concepts to learn — the two failures (must be kept separate):**

```text
Retrieval failure:  the relevant information was not retrieved, or was ranked too low.
Generation failure: the correct information was retrieved, but a future LLM step
                    answers incorrectly.
```

Generation is not implemented yet, so **this lab analyzes retrieval failures only**.
No LLM is needed for this analysis.

**Implementation tasks:**
- Create `evaluation/failures.md` from the template below, and fill in **at least one**
  real case (find it by scanning the labeled results for questions where the relevant chunk
  is missing or sits below rank 3).

```markdown
## Failure 1 — QXX
- Question: ...
- Expected/relevant information: ...
- Retrieved results: ... (top-5 with ranks, scores, page)
- Rank of the relevant chunk (if found): ...
- Was the correct chunk outside Top-K? yes/no
- Observed failure pattern: ... (e.g. "right page, wrong chunk", "synonyms not matched",
  "answer spread across chunks", "not answerable question retrieved random chunks")
- Suspected reason: ...
- Possible improvement: ... (e.g. smaller/larger chunks, hybrid search, reranking)
```

**Expected files/modules:** new `evaluation/failures.md`.

**Data/artifacts:** `evaluation/failures.md` with ≥1 complete failure entry.

**Acceptance criteria:**
- At least one failure entry with every template field filled.
- The failure is classified as a **retrieval failure**; any note about generation is clearly
  marked "out of scope (Generation not implemented)".

**Verification:** re-run `scripts/inspect_results.py` for the failing question and confirm
the documented ranks/scores match the persisted results.

**Dependencies:** 12.4/12.5.

---

#### Phase 12 — Definition of done

- `evaluation/dataset.json` exists with 20 questions (min 15), reused across experiments.
- Top-10 retrieval results persisted for the baseline configuration
  (`evaluation/results/baseline_500_50_top10.jsonl`), 200 records, full schema.
- All records labeled `relevant` / `not_relevant` (persisted, idempotent).
- Per-question and average **Precision@3** and **Precision@5** computed and saved.
- Top-3 vs Top-5 vs Top-10 analysis written for at least 3 questions.
- At least one real retrieval failure documented in `evaluation/failures.md`.
- **Zero LLM calls were made during this entire phase.**

---

### Phase 13 — Retrieval Optimization (the lab: improve with evidence)

**Status: NOT STARTED — depends on Phase 12 (baseline). No LLM, no UI.**

**Purpose:** improve retrieval by changing **one thing at a time** and measuring each change
against the Phase 12 baseline using the *same* evaluation questions and the *same* manual
labeling methodology. We compare at least two (preferably three) chunk configurations in
isolated vector stores, pick a final configuration based on measured results, and document
the justification. Advanced methods (keyword/hybrid/reranking) remain **optional**.

**Concepts to learn:** controlled experiments (one variable changed), experiment-specific
index isolation, metric comparison tables, reproducibility/regression checks, the
hypothesis→experiment→decision loop. **Golden rule: never change chunking, embeddings,
prompts, and retrieval at the same time.**

---

#### 13.1 Baseline retrieval configuration (frozen reference)

**Purpose:** pin the exact baseline so every change is a controlled diff.

**Implementation tasks:** document the baseline in `evaluation/experiment_notes.md`:

```text
Baseline (Phase 12):
  chunk_size = 500, chunk_overlap = 50
  embedding model = sentence-transformers/all-MiniLM-L6-v2 (unchanged across experiments)
  vector store = Chroma, collection "documents", L2 distance space
  default K = 5 (from .env TOP_K); evaluation uses K = 10
  retrieval method = similarity_search_with_score
```

**Expected files/modules:** `evaluation/experiment_notes.md` (new).

**Data/artifacts:** recorded baseline config + Phase 12 metrics (P@3, P@5).

**Acceptance criteria:** the baseline config and its P@3/P@5 numbers are written down before
any experiment runs.

**Verification:** re-read Phase 12 metric files and confirm they match the notes.

**Dependencies:** Phase 12.

---

#### 13.2 Chunk configuration experiments

**Purpose:** test whether chunk size/overlap changes retrieval quality, measured the same
way as the baseline.

**Concepts to learn:** chunk size tradeoffs (small = precise but fragmented; large =
contextual but noisy); keeping all other variables fixed.

**Implementation tasks:**
- Candidate configurations (treat as starting points, tune if justified):

```text
Baseline 500/50   (already measured in Phase 12)
Large    800/100
Small    300/50
```

- Extend `scripts/ingest.py` with CLI flags `--chunk-size`, `--chunk-overlap`,
  `--persist-dir` (defaults keep current behavior, so existing usage is unchanged).
- For each candidate: write chunks using the existing `chunk_documents(chunk_size,
  chunk_overlap)` and index into the configuration's own store (13.3).
- Run the Phase 12 evaluation runner against each store, with the **same** `dataset.json`
  and `top_k = 10`, then label and compute metrics exactly as in Phase 12.

**Expected files/modules:** modify `scripts/ingest.py`; new `scripts/run_experiment.py`
(a thin orchestrator: build store → run evaluation → point labeling/metrics at the result).

**Data/artifacts:** per-config results `evaluation/results/<config>_top10.jsonl`,
per-config labels, per-config `evaluation/metrics/<config>_p3.json` / `_p5.json`.

**Acceptance criteria:**
- At least 2 configurations compared (prefer 3: 500/50, 800/100, 300/50).
- Same questions, same embedding model, same retrieval method for every configuration.
- Chunk counts differ in the expected direction (large config → fewer chunks, small config →
  more chunks), confirming the splitter actually changed.

**Verification:** `python scripts/ingest.py --chunk-size 800 --chunk-overlap 100
--persist-dir data/chroma_db/experiments/large_800_100` then check chunk counts and run
evaluation for that config.

**Dependencies:** 13.1; Phase 12 tooling.

---

#### 13.3 Experiment-specific vector stores (isolation)

**Purpose:** guarantee each configuration is queried against its **own** index — a query must
never touch chunks created with a different chunk configuration.

**Concepts to learn:** why isolation matters for fairness; directory layout; keeping the
baseline development store untouched.

**Implementation tasks:**
- Layout:

```text
data/chroma_db/
  ...                       ← existing baseline store (leave untouched; rebuildable)
  experiments/
    baseline_500_50/        ← re-built copy of the baseline config for experiments
    large_800_100/
    small_300_50/
```

  Experiments always query only `data/chroma_db/experiments/<config>/`.
- Parameterize `src/vectorstore.py`:
  `get_vectorstore(persist_dir=CHROMA_DB_DIR)` and
  `add_documents(documents, persist_dir=...)` (defaults keep current behavior).
- Parameterize `src/retrieval.py`: `retrieve(query, top_k=TOP_K, persist_dir=...)`.
- Keep `collection_name="documents"` inside each directory — isolation comes from the
  separate persist directories, not from collection names.

**Expected files/modules:** modify `src/vectorstore.py`, `src/retrieval.py`;
`scripts/run_experiment.py` passes `--persist-dir`.

**Data/artifacts:** one Chroma store per configuration under
`data/chroma_db/experiments/<config>/`. Extend `.gitignore` to ignore
`data/chroma_db/experiments/` (rebuildable artifacts).

**Acceptance criteria:**
- Querying store A returns only chunks that store A contains (verify by comparing chunk
  counts per store).
- The original `data/chroma_db/` store is never modified by experiments.

**Verification:** list chunk counts per store (`store.get()["ids"]`) and confirm the small
config store has more chunks than the large config store.

**Dependencies:** 13.2.

---

#### 13.4 Metric comparison

**Purpose:** compare configurations on the same questions with the same methodology, using
measured numbers — never intuition.

**Implementation tasks:**
- Create `scripts/compare_configs.py` that reads each config's labeled results and
  `metrics.py` output, and prints a table:

```text
Configuration        P@3 avg   P@5 avg
baseline_500_50      0.733     0.620
large_800_100        0.600     0.540
small_300_50         0.800     0.680
```

- Persist `evaluation/metrics/comparison.md` including the per-question P@3/P@5 tables and
  the failure-analysis differences (which failure cases are fixed/unchanged/new).

**Expected files/modules:** new `scripts/compare_configs.py` (reuses `src/metrics.py`).

**Data/artifacts:** `evaluation/metrics/comparison.md`.

**Acceptance criteria:**
- Table shows P@3 and P@5 averages for every configuration tested (≥2, prefer 3).
- Conclusion is stated as numbers + qualitative failure analysis (from `evaluation/failures.md`),
  not "looks better".

**Verification:** re-run `scripts/compare_configs.py`; confirm numbers match each config's
metric files.

**Dependencies:** 13.2/13.3 (results + labels for each config).

---

#### 13.5 Optional advanced retrieval experiment (keyword / hybrid / reranking)

**Purpose:** OPTIONAL — test one advanced retrieval method against the vector baseline. Not a
prerequisite for completing the lab; the core plan remains valid without it.

**Concepts to learn:** keyword search (BM25) for exact terms; hybrid search (vector +
keyword combined); reranking (cross-encoder re-scoring top-20 candidates); how each would
address specific failure patterns from 12.9.

**Implementation tasks (only if pursued):**
- Keyword: BM25 over the same chunks (e.g. `rank_bm25`) — score and rank top-10.
- Hybrid: combine vector distance and BM25 scores (normalized) into one ranking.
- Rerank: retrieve top-20 with vectors, then re-rank with a cross-encoder, keep top-10.
- For whichever method is chosen, produce results in the **same record schema**, label them
  with the same `scripts/label_results.py`, and add the config to the `compare_configs.py`
  table. Store results under `evaluation/results/experiments/<method>_top10.jsonl`.
- Optionally note a future **section-aware chunking** idea here: heading-based `section`
  metadata added during chunking (would require re-indexing a configuration's store).

**Expected files/modules:** new `scripts/experiment_hybrid.py` (or `_bm25.py` / `_rerank.py`)
only if the experiment is done.

**Data/artifacts:** optional `evaluation/results/experiments/<method>_top10.jsonl` + labels +
metrics row.

**Acceptance criteria:** if done, the method appears in the comparison table with the same
metrics; if not done, the core Phase 13 deliverables (13.6–13.8) must still be complete.

**Verification:** same as 13.4.

**Dependencies:** 13.4; 12.9 failure patterns to motivate the choice.

---

#### 13.6 Final configuration selection

**Purpose:** choose **one** final retrieval configuration from the measured evidence.

**Implementation tasks:** after experiments, document the final choice in
`experiments/final_configuration.md` covering **all** of:

```text
chunk size:
chunk overlap:
embedding model:
vector store:
similarity/distance metric:
default K:
optional retrieval method (if any):
```

**Expected files/modules:** new `experiments/final_configuration.md`.

**Data/artifacts:** the written final configuration.

**Acceptance criteria:** the choice is derived from the comparison table and failure
analysis; every field above is filled in; no field is "to be decided".

**Verification:** each field maps to an actual value used in a tested configuration.

**Dependencies:** 13.4.

---

#### 13.7 Final configuration justification

**Purpose:** explain *why* the configuration was chosen in one concise engineering
justification — backed by measured results.

**Implementation tasks:** append to `experiments/final_configuration.md` a short
justification that cites specific numbers, e.g.:

```text
Final: small_300_50 because P@3 = 0.800 (vs baseline 0.733) and P@5 = 0.680
(vs baseline 0.620); failure analysis shows Failure 1 fixed because the answer
now fits within one chunk. The 3→5→10 noise analysis shows a slower precision
drop, so top-5 context stays clean.
```

**Expected files/modules:** `experiments/final_configuration.md` (appended).

**Data/artifacts:** the justification text.

**Acceptance criteria:** the justification references at least one measured metric and at
least one failure-analysis finding; no unsupported claims ("feels better").

**Verification:** every cited number can be found in `evaluation/metrics/` files.

**Dependencies:** 13.6.

---

#### 13.8 Reproducibility / regression evaluation

**Purpose:** prove the final configuration is reproducible and lock it in as the new default
so future changes are measured against it.

**Implementation tasks:**
- Update `src/config.py` defaults (e.g. `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`) to the final
  values, and mirror them in `.env.example`.
- Add a reproducible command sequence (documented in `evaluation/experiment_notes.md`, or a
  `scripts/rerun_evaluation.sh`): build final store → run evaluation → label → metrics.
- Run a fresh regression evaluation from a clean store; recorded metrics must match the
  saved `evaluation/metrics/` values.

**Expected files/modules:** modify `src/config.py`, `.env.example`; optional
`scripts/rerun_evaluation.sh`.

**Data/artifacts:** a fresh run of `evaluation/results/final_<config>_top10.jsonl` and
matching metrics.

**Acceptance criteria:** a from-scratch rebuild of the final store reproduces the recorded
P@3/P@5 (within floating-point tolerance); config defaults match the final choice.

**Verification:** delete/re-create `data/chroma_db/experiments/<final>` and re-run the full
sequence; compare metrics.

**Dependencies:** 13.6/13.7.

---

#### Phase 13 — Definition of done

- Baseline vs at least one alternative chunk configuration (prefer three) compared with the
  **same** 20 questions, same embedding model, same retrieval method.
- Each configuration has its **own isolated** Chroma store under
  `data/chroma_db/experiments/<config>/`.
- A metric comparison table (`evaluation/metrics/comparison.md`) with P@3 and P@5.
- A final configuration selected and justified from measured results
  (`experiments/final_configuration.md`).
- A regression run proves the final configuration is reproducible.
- Any optional advanced method (13.5) is clearly additive — the core lab is complete without
  it.

---

## 10. Evaluation strategy (the single source of truth)

**Primary lab methodology — manual relevance labeling + Precision@K (Phases 12–13, no LLM):**
- **Fixed evaluation dataset** (`evaluation/dataset.json`): 20 questions (min 15) covering
  different information types in the PDF; reused by **every** experiment.
- **Every question is run through retrieval** with `top_k = 10`; the full top-10 is persisted
  per configuration as JSONL (`evaluation/results/<config>_top10.jsonl`).
- **A human labels each retrieved chunk** `relevant` / `not_relevant`; the label is stored
  inside the persisted record (attached to question + chunk + rank).
- **Precision@K** = (number of relevant chunks in top K) / K. Reported per-question and
  averaged, for **K = 3** and **K = 5**.
- **Top-K analysis** compares Top-3 vs Top-5 vs Top-10 to expose the useful-context vs noise
  tradeoff.
- **Failure analysis** documents at least one real retrieval failure (relevant info not
  retrieved or ranked too low).
- **Score semantics:** Chroma returns **L2 distance** (lower = more similar); rank 1 =
  smallest score. Never read a single score as "similarity" without checking the sign.

**Complementary measures (optional / later):**
- **HitRate@K** = fraction of questions where at least one *expected* chunk is in top K, and
  **Recall@K** = fraction of expected chunks found — these need `expected_chunk_ids` added to
  `dataset.json`; useful as a cross-check but not required by the lab.
- **Generation checks** (groundedness, correctness, refusal, citation accuracy) apply only
  after Generation is implemented (Phase 8) — out of scope for the retrieval lab.

**Framing rule for every bug:** ask "did the right chunk get retrieved?" (retrieval) before
"did the LLM answer correctly?" (generation). This single question prevents most debugging
spiral. In the retrieval lab, only the first question is in scope.

---

## 11. Common mistakes and failure modes (quick reference)

| Failure | Symptom | Most likely cause | Where to fix |
|---|---|---|---|
| Empty/garbled PDF text | pages exist, text is blank | scanned PDF (no text layer) / OCR needed | Phase 3 |
| Bad retrieval | right answer but wrong chunks shown | bad chunking, wrong embedding model, K too small | Phases 4/5/7 → 12–13 |
| Right chunk ranked low | it exists but not in top-K | chunk boundary cuts the answer, K too small | Phase 12 (document it) → 13 (chunk experiments; optional rerank 13.5) |
| Relevant chunk missing entirely | not found in top-10 | retrieval failure | Phase 12.9 → 13 |
| Scores look inverted | "similarity" gets smaller | Chroma returns **L2 distance**, lower = better | Phases 6/12.2 |
| Evaluation not reproducible | numbers change every run | different questions/stores between runs | Phases 12.1/12.3 |
| LLM ignores context | confident wrong answer | weak prompt, too much noise in context | Phase 8 (prompt design) |
| Hallucination on missing info | invented facts | no refusal instruction in prompt | Phase 8 |
| Duplicated data | same chunk retrieved many times / stale answers | re-indexing without clearing the store; querying wrong experiment store | Phase 6 / 13.3 |
| Citations wrong | [1] points at wrong page | metadata lost, marker resolution bug | Phase 9 |
| UI breaks but CLI works | only Gradio fails | RAG logic leaked into the UI | Phase 11 |

---

## 12. Advanced improvement roadmap (after baseline)

The retrieval lab (Phase 13) covers **chunk configuration experiments** as the primary
optimization, plus **optional** keyword / hybrid / reranking experiments (13.5). Ideas beyond
the lab's scope — better embedding models, query rewriting, metadata filtering,
multi-document retrieval, context compression, advanced evaluation frameworks (e.g. `ragas`),
streaming — remain future work, **always measured against the Phase 12 baseline, one change
at a time**. Never change chunking, embeddings, retrieval, and (later) prompts in the same
experiment.

---

## 13. Phase completion format (used at the end of every executed phase)

```text
Phase: X

What we built:
...

What I learned:
...

Files created/changed:
...

How to test:
...

Expected result:
...

Common problems:
...

Definition of done:
...

Next phase:
...
```

We do **not** start the next phase until the current one is verified.

---

## 14. Ready to begin

**Current state:** Phases 2–7 are DONE (Setup, Ingestion, Chunking, Embeddings, Vector
store, Retrieval). The next work is the **Retrieval Evaluation lab (Phase 12)**, immediately
followed by **Retrieval Optimization (Phase 13)** — see the `Next Implementation Sequence`
section below. Phases 8–11 (Generation, Sources, Pipeline, Gradio) come afterwards.

Wait for my approval before starting implementation.

---

## 15. Next Implementation Sequence

Exact order for the retrieval evaluation / optimization lab. Each step depends only on the
previous one and on Phases 2–7. It is designed to complete the lab checklist incrementally
with minimal rework.

1. **12.1 — Evaluation dataset.** Write `evaluation/dataset.json` (20 questions, IDs `Q01`…).
   Verify every question (except the deliberate negative one) is answerable from the PDF.
2. **12.2/12.3 — Runner + persistence.** Add eval paths to `src/config.py`; create
   `src/evaluation.py` and `scripts/run_evaluation.py`; persist
   `evaluation/results/baseline_500_50_top10.jsonl` (200 records).
3. **12.4 — Inspection.** Create `scripts/inspect_results.py`; confirm it prints all fields
   (including `section: N/A`) for any question.
4. **12.5 — Manual labeling.** Create `scripts/label_results.py`; label the baseline file
   (`relevant` / `not_relevant`, persisted, idempotent).
5. **12.6/12.7 — Metrics.** Create `src/metrics.py` and `scripts/metrics.py`; produce
   per-question + average Precision@3 and Precision@5.
6. **12.8 — Top-K analysis.** Create `scripts/analyze_topk.py`; write the 3-vs-5-vs-10
   interpretation for at least 3 questions.
7. **12.9 — Failure analysis.** Write `evaluation/failures.md` with ≥1 real retrieval
   failure. Phase 12 is now complete (baseline measured).
8. **13.1 — Baseline config.** Record the frozen baseline in `evaluation/experiment_notes.md`.
9. **13.2/13.3 — Experiments + isolation.** Parameterize `src/vectorstore.py` /
   `src/retrieval.py`; extend `scripts/ingest.py` flags; add `scripts/run_experiment.py`;
   build `data/chroma_db/experiments/{baseline_500_50,large_800_100,small_300_50}`. Run
   evaluation + labeling + metrics for each config (reusing steps 2–5).
10. **13.4 — Comparison.** Create `scripts/compare_configs.py`;
    write `evaluation/metrics/comparison.md` (P@3/P@5 table + failure-diff analysis).
11. **13.5 — Optional advanced experiment** (keyword/hybrid/rerank) — only if pursued; core
    plan remains valid without it.
12. **13.6/13.7 — Final configuration + justification.** Write
    `experiments/final_configuration.md` (all fields + one evidence-based justification).
13. **13.8 — Reproducibility/regression.** Update `src/config.py` / `.env.example` to the
    final values; add the re-run sequence; verify a clean rebuild reproduces the recorded
    metrics.

After the lab: **Phase 8 — Generation**, then Phases 9–11, then any post-Generation
evaluation from the complementary measures in section 10.