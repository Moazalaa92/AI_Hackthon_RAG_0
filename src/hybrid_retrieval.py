"""Hybrid retrieval: dense (Chroma/L2) + lexical (BM25) + Reciprocal Rank Fusion.

Single-variable experiment layer on top of the existing dense pipeline.
Dense retrieval reuses the exact same 800/100 + MiniLM store and the same
`similarity_search_with_score` call as the dense-only control. BM25 is built
from the SAME stored chunks (read directly from the Chroma collection, so
there is no re-chunking and no re-embedding), guaranteeing both retrievers
operate over identical chunk units joinable by document_id.

Fusion: Reciprocal Rank Fusion (RRF), k = 60 (standard default, not tuned on
the evaluation set — the 20-question set is too small for weight tuning).

    RRF_score(d) = sum over retrievers r of 1 / (k + rank_r(d))

Only retrievers that returned the chunk in their candidate list contribute a
term, so a chunk found by both retrievers scores strictly higher than a chunk
found by either alone. No reranking model, no learned weights.
"""

import re

import chromadb

from src.config import EVAL_TOP_K
from src.embeddings import DEFAULT_EMBEDDING_MODEL
from src.vectorstore import get_vectorstore

DENSE_CANDIDATES = 20
BM25_CANDIDATES = 20
RRF_K = 60

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text):
    """Simple, transparent tokenizer: lowercase alphanumeric tokens.

    Keeps numbers, acronyms (MRI, EEG) and drug names (topiramate,
    ethosuximide, valproate) intact as single tokens, which is what the
    lexical arm is meant to contribute.
    """
    return _TOKEN_RE.findall(text.lower())


def build_bm25_index(persist_dir):
    """Build a BM25Okapi index over the chunks stored in a vector store.

    Reads stored chunk texts and metadata directly from the Chroma collection
    (no embedding function needed to read persisted text). Returns
    (bm25, doc_ids, chunk_info) where doc_ids[n] is the document_id in the
    same order as the BM25 corpus, and chunk_info maps document_id to
    {"text": ..., "metadata": ...}.
    """
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection("documents")
    data = collection.get(include=["documents", "metadatas"])

    chunk_info = {}
    corpus = []
    for doc_id, text, metadata in zip(
        data["ids"], data["documents"], data["metadatas"]
    ):
        chunk_info[doc_id] = {"text": text, "metadata": metadata}
        corpus.append(tokenize(text))

    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(corpus)
    return bm25, data["ids"], chunk_info


def _fusion_scores(union_ids, dense_rank, bm25_rank, rrf_k=RRF_K):
    scores = {}
    for doc_id in union_ids:
        score = 0.0
        if doc_id in dense_rank:
            score += 1.0 / (rrf_k + dense_rank[doc_id]["rank"])
        if doc_id in bm25_rank:
            score += 1.0 / (rrf_k + bm25_rank[doc_id]["rank"])
        scores[doc_id] = score
    return scores


def _candidate_records(question, vectorstore, bm25, chunk_info,
                       dense_k=DENSE_CANDIDATES, bm25_k=BM25_CANDIDATES,
                       rrf_k=RRF_K):
    """Run dense + BM25, union the candidates, fuse with RRF.

    Returns a list of candidate records ranked by fusion score (rank = 1..n,
    where n is the size of the union), each carrying retrieval provenance:
    retrieved_by, dense_rank, bm25_rank, dense_score, bm25_score, fusion_score.
    """
    q = question["question"]

    dense_rank = {}
    dense_hits = vectorstore.similarity_search_with_score(q, k=dense_k)
    for rank, (doc, score) in enumerate(dense_hits, start=1):
        dense_rank[doc.id] = {"rank": rank, "score": float(score)}

    bm25_rank = {}
    scored = sorted(
        zip(chunk_info.keys(), bm25.get_scores(tokenize(q))),
        key=lambda item: -item[1],
    )
    for rank, (doc_id, score) in enumerate(scored[:bm25_k], start=1):
        bm25_rank[doc_id] = {"rank": rank, "score": float(score)}

    union_ids = set(dense_rank) | set(bm25_rank)
    fusion = _fusion_scores(union_ids, dense_rank, bm25_rank, rrf_k=rrf_k)

    inf = 10**9
    fused_ids = sorted(
        union_ids,
        key=lambda d: (
            -fusion[d],
            dense_rank.get(d, {}).get("rank", inf),
            bm25_rank.get(d, {}).get("rank", inf),
            d,
        ),
    )

    records = []
    for rank, doc_id in enumerate(fused_ids, start=1):
        metadata = chunk_info[doc_id]["metadata"]
        dense_entry = dense_rank.get(doc_id)
        bm25_entry = bm25_rank.get(doc_id)
        retrieved_by = []
        if dense_entry is not None:
            retrieved_by.append("dense")
        if bm25_entry is not None:
            retrieved_by.append("bm25")

        records.append(
            {
                "question_id": question["id"],
                "question_text": q,
                "rank": rank,
                "score": round(fusion[doc_id], 6),
                "chunk_id": metadata.get("chunk_id"),
                "document_id": doc_id,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "page_label": metadata.get("page_label"),
                "section": metadata.get("section"),
                "chunk_text": chunk_info[doc_id]["text"],
                "relevant": None,
                "retrieved_by": "+".join(retrieved_by),
                "dense_rank": dense_entry["rank"] if dense_entry else None,
                "bm25_rank": bm25_entry["rank"] if bm25_entry else None,
                "dense_score": dense_entry["score"] if dense_entry else None,
                "bm25_score": bm25_entry["score"] if bm25_entry else None,
                "fusion_score": round(fusion[doc_id], 6),
            }
        )
    return records


def run_hybrid(dataset, persist_dir, model_name=DEFAULT_EMBEDDING_MODEL,
               dense_k=DENSE_CANDIDATES, bm25_k=BM25_CANDIDATES,
               final_k=EVAL_TOP_K, rrf_k=RRF_K):
    """Run hybrid retrieval for every dataset question.

    Returns (topk_records, candidate_records):
      topk_records       - final Top-K after fusion (rank 1..final_k), the
                           frozen file for the existing evaluation pipeline.
      candidate_records  - every union candidate ranked by fusion (rank 1..n),
                           used for candidate-recall analysis. Top-K records
                           are exactly candidate records with rank <= final_k.
    `score` is the RRF fusion score (higher = better), unlike the dense-only
    `score` (L2 distance, lower = better). Raw dense distance is preserved in
    `dense_score` for provenance.
    """
    vectorstore = get_vectorstore(persist_dir, model_name=model_name)
    bm25, _, chunk_info = build_bm25_index(persist_dir)

    candidates = []
    for question in dataset["questions"]:
        candidates.extend(
            _candidate_records(
                question,
                vectorstore,
                bm25,
                chunk_info,
                dense_k=dense_k,
                bm25_k=bm25_k,
                rrf_k=rrf_k,
            )
        )

    topk = [r for r in candidates if r["rank"] <= final_k]
    return topk, candidates