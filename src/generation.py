"""Generation: question + retrieved chunks -> grounded LLM answer (Phase 8).

This module is ONLY the generation stage. It never performs retrieval; it
receives already-retrieved chunks (as produced by the frozen retrieval
pipeline) and turns them into a grounded natural-language answer.

    retrieval(question) -> retrieved_chunks
    generate_answer(question, retrieved_chunks) -> answer

Chunk metadata (chunk_id, page, page_label, source, rank, ...) is preserved
through context construction so the answer-to-chunk relationship survives for
the future Sources/citations phase.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

GROUNDING_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
1. Ground every claim in the provided context only. Do not use outside
   knowledge and do not invent missing information.
2. Answer the exact question asked. When the context contains several closely
   related recommendations, distinguish them and answer the one the question
   asks about — do not substitute a nearby recommendation merely because it is
   semantically similar.
3. When the question asks for multiple items, conditions, criteria, steps or
   components, include the important components that the context supports, and
   do not omit important qualifying conditions.
4. Only refuse when the context genuinely does not contain enough information
   to answer. If the context directly contains the answer, answer from that
   evidence even if the document adds no further explanation.
5. Be concise and directly relevant. Do not add tangential information just
   because it appears in the context.
"""


def build_context(chunks):
    """Convert retrieved chunks into an LLM-readable context string.

    `chunks` is a list of records as produced by the retrieval pipeline, each
    a dict with at least a text field (`chunk_text` or `text`/`page_content`)
    plus metadata (`chunk_id`, `page`, `page_label`, `source`, `rank`, ...).

    Each chunk becomes:

        [Chunk N] (chunk_id=..., page=..., source=..., rank=...)
        <chunk text>

    The metadata is kept in the context so the answer can later be mapped back
    to the exact source chunk. The original chunk records are not modified.
    """
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.get("chunk_text") or chunk.get("text") or chunk.get("page_content") or ""
        meta_parts = []
        for key in ("chunk_id", "document_id", "page", "page_label", "source", "rank", "reranker_rank", "retrieved_by"):
            if chunk.get(key) is not None:
                meta_parts.append(f"{key}={chunk[key]}")
        meta = ", ".join(meta_parts)
        block = f"[Chunk {index}] ({meta})\n{text}" if meta else f"[Chunk {index}]\n{text}"
        blocks.append(block)
    return "\n\n".join(blocks)


def build_prompt(question, context):
    """Return the grounding prompt messages (system + user)."""
    system = SystemMessage(content=GROUNDING_SYSTEM_PROMPT)
    user = HumanMessage(
        content=(
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Answer the question using only the provided context. If the context does "
            "not contain enough information to answer the question, say that the answer "
            "was not found in the provided documents."
        )
    )
    return [system, user]


def generate_answer(question, retrieved_chunks, max_tokens=600):
    """Generate a grounded answer for `question` from `retrieved_chunks`.

    Args:
        question: the question string.
        retrieved_chunks: list of chunk records from the retrieval pipeline
            (each a dict with chunk text + metadata).
        max_tokens: maximum tokens for the model response.

    Returns:
        The generated answer text.

    Raises:
        ValueError: if `question` is empty.
    """
    if not question or not str(question).strip():
        raise ValueError("question must be a non-empty string")

    context = build_context(retrieved_chunks)
    messages = build_prompt(str(question).strip(), context)

    client = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        max_tokens=max_tokens,
    )
    response = client.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)
    return str(content)