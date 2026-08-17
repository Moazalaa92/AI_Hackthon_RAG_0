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
| `requirements.txt` | `chromadb`, `gradio`, `openai`, `pypdf`, `python-dotenv`        | Will be extended per phase        |
| `data/pdfs/`       | `Project_pdf.pdf` — our first test document                     | Use as the baseline test PDF      |
| `.venv/`           | Virtual environment                                             | Keep using it                     |
| `.gitignore`       | Ignores `.venv`, `.env`, `__pycache__`, `data/extracted/`       | Extend later for `data/chroma_db/` |

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
| **Similarity score** | How close two vectors are (we will use cosine similarity). |
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
│   └── chroma/                   ← persisted vector store (created by us)
│
├── src/
│   ├── __init__.py
│   ├── config.py                 ← reads .env, holds model/embedding settings
│   ├── ingestion.py              ← PDF → Documents (PyPDFLoader)
│   ├── chunking.py               ← Documents → Chunks (splitter + metadata)
│   ├── embeddings.py             ← embedding model setup (all-MiniLM-L6-v2)
│   ├── vectorstore.py            ← Chroma setup: add chunks, persist
│   ├── retrieval.py              ← question → top-K chunks (observable)
│   ├── generation.py             ← context + question → LLM answer
│   ├── sources.py                ← citation resolution (marker → file/page/chunk)
│   └── pipeline.py               ← rag.query(question) → Answer + Sources
│
├── scripts/
│   ├── ingest.py                 ← CLI: index PDFs into Chroma
│   ├── inspect.py                ← CLI: inspect extracted pages / chunks
│   ├── retrieve.py               ← CLI: retrieval-only inspection (no LLM)
│   ├── ask.py                    ← CLI: full RAG question → answer + sources
│   └── evaluate.py               ← CLI: run the evaluation dataset, print metrics
│
├── evaluation/
│   └── dataset.json              ← curated questions + expected chunks/answers
│
├── app/
│   └── gradio_app.py             ← thin UI calling pipeline.py
│
├── experiments/                  ← scratch notes and comparisons (not shipped)
└── tests/                        ← small sanity tests (pytest), optional but useful
```

**Why this shape (and what we dropped from the suggested structure):**
- Dropped `src/ingestion/`, `src/processing/`... sub-packages → single modules per concern.
  Simpler to read, fewer files, same boundaries.
- Dropped `tests/` until Phase 10+ — tests are valuable, but early phases use *inspection
  scripts* as verification, which is more instructive for a learner.
- Kept `experiments/` — this is where we record results of each improvement so we never lose
  track of "did the change help?"

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

---

### Phase 1 — RAG architecture fundamentals (no code)

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

**Purpose:** persist chunks + embeddings + metadata, then search them.

**Concepts to learn:** what a vector store stores (embedding vector, original text, metadata),
why we store the original text (the LLM needs words, not vectors; and the vector alone cannot
be shown to a human), why metadata, how similarity search works (embed query → compare against
every stored vector → return closest), what Top-K means, what the returned score means
(cosine distance in Chroma: lower = more similar).

**Build:** `src/vectorstore.py` with `Chroma(persist_directory="data/chroma_db/")`. Add chunks
with embeddings + metadata.

**Under the hood:** Chroma persists to SQLite + parquet files in the folder. It computes
cosine distance between the query vector and all stored vectors and ranks them.

**Verification:** insert `Project_pdf.pdf` chunks, then run a tiny test that searches for
several terms and prints results with scores. Confirm persistence by re-opening the store in
a fresh process and searching again.

**Common mistakes:** re-indexing duplicates (add without clearing); forgetting persistence
directory; confusing the score (Chroma returns *distance*).

**Definition of done:** you can add to and search Chroma directly, explain Top-K and scores,
and see data persist across runs.

---

### Phase 7 — Retrieval (the heart of the system)

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

### Phase 12 — Baseline evaluation

**Purpose:** measure the system *before* improving it — the reference point for everything
after.

**Concepts to learn:** an evaluation dataset (curated, manually checked), `HitRate@K`
("was the expected chunk in the top K?"), `Recall@K` ("fraction of expected chunks found"),
and generation checks (groundedness, correctness, refusal, citation accuracy). Retrieval and
generation are measured **separately**.

**Build:**
- `evaluation/dataset.json`: ~10–20 questions for `Project_pdf.pdf`, each with expected
  chunks/pages, expected facts, and answerable/not-answerable flag.
- `scripts/evaluate.py`: runs every question through retrieval-only (metrics @1/@3/@5/@10)
  and through full RAG (generation checks), printing a metrics table.

**Verification:** the script prints baseline numbers; you manually spot-check a sample of
answers to confirm the metrics agree with reality.

**Common mistakes:** judging with vague "looks good"; mixing retrieval and generation into one
number; a dataset too small or too easy.

**Definition of done:** you have written, measurable baseline numbers for retrieval and
generation. From now on, every change is compared to these.

---

### Phase 13 — Improvements (one at a time, measured)

**Purpose:** learn how to improve the system with evidence, never by vibes.

**Rule: one change at a time.** Baseline → one change → same evaluation → compare → keep or
reject. Candidate improvements, in recommended order:

1. **Chunking experiments** — tune size/overlap, sentence-aware splitting, drop tiny/noisy
   chunks. Cheap and often fixes retrieval failures.
2. **Better embedding models** — swap `all-MiniLM-L6-v2` for a stronger model (e.g.
   `BAAI/bge-small-en-v1.5` or `bge-m3`) and re-run the same eval. Measure only the model
   change.
3. **Reranking** — keep vector search for candidates, then re-rank top ~20 with a
   cross-encoder; fixes "right chunk retrieved but ranked too low".
4. **Hybrid search** — combine vector + keyword (BM25) search for questions where exact terms
   matter.
5. **Query rewriting** — reformulate the user question before embedding to improve retrieval.
6. **Metadata filtering** — retrieve only from a chosen document/page.
7. **Multi-document retrieval** — index many PDFs, verify sources stay correct.
8. **Context compression** — drop irrelevant chunks/passages before generation to reduce
   noise.
9. **Advanced evaluation frameworks** — only now, if needed (e.g. `ragas`).
10. **Streaming** — stream LLM tokens into Gradio for a better UX.

**Golden rule reminder:** never change chunking, embeddings, reranking, prompts, and
retrieval at the same time — otherwise you cannot know what improved anything.

---

## 10. Evaluation strategy (the single source of truth)

**Retrieval metrics (Phase 7 script):**
- **HitRate@K** = fraction of questions where at least one expected chunk is in the top K.
- **Recall@K** = fraction of the expected chunks found within the top K.
- Compare @1, @3, @5, @10 to see how K changes behavior.

**Generation checks (Phase 12 script):**
- **Groundedness** — is the answer supported by the retrieved context?
- **Correctness** — does the answer contain the expected facts?
- **Refusal behavior** — does it correctly refuse unanswerable questions?
- **Citation accuracy** — do the citations point at the right chunks?

**Framing rule for every bug:** ask "did the right chunk get retrieved?" (retrieval) before
"did the LLM answer correctly?" (generation). This single question prevents most debugging
spiral.

---

## 11. Common mistakes and failure modes (quick reference)

| Failure | Symptom | Most likely cause | Where to fix |
|---|---|---|---|
| Empty/garbled PDF text | pages exist, text is blank | scanned PDF (no text layer) / OCR needed | Phase 3 |
| Bad retrieval | right answer but wrong chunks shown | bad chunking, wrong embedding model, K too small | Phases 4/5/7 |
| Right chunk ranked low | it exists but not in top-K | chunk boundary cuts the answer, K too small | Phase 4/7 → 13 (rerank) |
| LLM ignores context | confident wrong answer | weak prompt, too much noise in context | Phase 8 → 13 (compression) |
| Hallucination on missing info | invented facts | no refusal instruction in prompt | Phase 8 |
| Duplicated data | same chunk retrieved many times / stale answers | re-indexing without clearing the store | Phase 6 |
| Citations wrong | [1] points at wrong page | metadata lost, marker resolution bug | Phase 9 |
| UI breaks but CLI works | only Gradio fails | RAG logic leaked into the UI | Phase 11 |

---

## 12. Advanced improvement roadmap (after baseline)

See Phase 13 for the full list. In short: chunking → embeddings → reranking → hybrid search →
query rewriting → metadata filtering → multi-doc → context compression → advanced evaluation →
streaming. **Always measured against the Phase 12 baseline, one change at a time.**

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

**Phase 1 is ready to begin.**

Wait for my approval before starting implementation.