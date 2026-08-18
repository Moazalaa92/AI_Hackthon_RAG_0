# Hackathon Lectures — RAG over a NICE Epilepsy Guideline (NG217)

A LangChain-based **retrieval-augmented generation (RAG)** project that answers questions
from `data/pdfs/Project_pdf.pdf` (NICE NG217, "Epilepsies in children, young people and
adults", 161 pages).

## Project status

| Component | Status |
|---|---|
| **RAG retrieval** | **VALIDATED / FROZEN FOR NOW** |
| **Generation** | NOT IMPLEMENTED |
| **Sources / citations** | NOT IMPLEMENTED |
| **End-to-end pipeline** | NOT IMPLEMENTED |
| **UI** | NOT IMPLEMENTED |

This is **not** yet a complete RAG QA system — retrieval is built and validated; the
Generation phase (Phase 8) is the next engineering step. See `PLAN.md` for the full plan and
phase status.

## Validated retrieval architecture (frozen)

```text
PDF → ingestion → chunking (800/100) → MiniLM embeddings
   → dense top-20 (Chroma/L2)  +  BM25 top-20  → union
   → cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) → final Top-10
```

Validated metrics (LLM-as-a-Judge labels; proxy for relevance, not clinical ground truth):

```text
Original 20-question benchmark:  P@3 0.5167, P@5 0.4000, Hit@3 0.90, Hit@5 0.95, Hit@10 0.95
27-question holdout:              P@3 0.5802, P@5 0.4519, Hit@3 0.9259, Hit@5 0.9259, Hit@10 0.9259
Answerable candidate recall:      19/19 benchmark, 25/25 holdout (100%)
```

Metrics are **retrieval quality**, not answer accuracy. Details:
`evaluation/metrics/comparison.md` (experiment history + conclusions),
`evaluation/experiment_notes.md` (hypothesis → decision log),
`evaluation/failures.md` (failure analysis).

## Repository layout

```text
src/            core modules (ingestion, chunking, embeddings, vectorstore, retrieval,
                evaluation, metrics, judge, hybrid_retrieval, reranking)
scripts/        CLIs (ingest, retrieve, run_evaluation, label_results, metrics,
                analyze_topk, hybrid/rerank runners and analysis, inspection)
evaluation/     dataset.json (20 benchmark Q), holdout_dataset_v1.json (27 Q),
                results/ (frozen JSONL per configuration + labels), metrics/ (JSON + comparison.md)
data/pdfs/      Project_pdf.pdf (source corpus)
data/chroma_db/ baseline store + experiments/ (isolated per-config stores)
app.py          legacy end-to-end demo (NOT canonical)
```

See `PLAN.md` for the authoritative phase-status table and source/script inventory.

## Quick usage

```bash
# Inspect retrieval (no LLM)
python scripts/retrieve.py "Which medicine is offered as first-line treatment for absence seizures?"

# Re-run the benchmark evaluation against a store (no LLM)
python scripts/run_evaluation.py --store data/chroma_db/experiments/large_800_100

# Label frozen results with the LLM judge (requires a working OpenRouter key in .env)
python scripts/label_results.py evaluation/results/reranked_hybrid_800_100_top10.jsonl

# Metrics on a labeled file
python scripts/metrics.py evaluation/results/reranked_hybrid_800_100_top10_labeled.jsonl
python scripts/analyze_topk.py evaluation/results/reranked_hybrid_800_100_top10_labeled.jsonl
```

Environment: copy `.env.example` to `.env` and set `LLM_API_KEY`, `LLM_BASE_URL`,
`LLM_MODEL` (OpenRouter / deepseek). `.env` is gitignored — never commit it.

## Dependencies

See `requirements.txt`: `python-dotenv`, `pypdf`, `langchain*` (community,
text-splitters, huggingface, chroma, openai), `sentence-transformers`, `chromadb`,
`gradio`, `rank_bm25` (added for the hybrid retrieval experiment).

## Evaluation limitations

- Relevance labels come from an **LLM judge** (`deepseek/deepseek-v4-flash`, temp 0), not
  clinically validated ground truth — engineering evidence, not clinical validation.
- Small evaluation sets: 20 benchmark + 27 holdout questions; Wilson 95% CIs are wide.
- Some judge-label variance observed across repeated runs (~5–7% label flips).
- Q20 / H26 / H27 are intentional negative cases (dosing/cost not in the guideline); do not
  interpret them as ordinary retrieval failures.
- P@3 / Hit@K / candidate recall are retrieval metrics — not answer accuracy.