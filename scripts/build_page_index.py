"""Build a resumable, deterministic Pixel RAG page index.

This opt-in CLI renders the frozen PDF and embeds pages only; it does not
change text retrieval, call an LLM, or create labels. Checkpoints are saved
every 20 pages so interrupted CPU jobs resume without re-embedding completed
pages.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.page_images import (
    DEFAULT_DPI,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PDF,
    render_pdf_pages,
    verify_page_label_mapping,
)
from src.visual_retrieval import (
    DEFAULT_INDEX,
    VISUAL_MODEL,
    embed_page_images,
    save_index,
)

CHECKPOINT_INTERVAL = 20


def main():
    parser = argparse.ArgumentParser(description="Build the Pixel RAG page index.")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--model", default=VISUAL_MODEL)
    parser.add_argument("--out", default=str(DEFAULT_INDEX))
    args = parser.parse_args()

    checked = verify_page_label_mapping()
    print(f"Verified page_label == str(page + 1) across {checked} labeled records.")
    records = render_pdf_pages(args.pdf, DEFAULT_OUTPUT_DIR, args.dpi)
    output_path = Path(args.out)
    checkpoint_path = output_path.with_name(f"{output_path.stem}.checkpoint.npz")

    completed = {}
    if checkpoint_path.exists():
        from src.visual_retrieval import load_index

        checkpoint = load_index(checkpoint_path)
        if checkpoint.model != args.model or checkpoint.dpi != args.dpi:
            raise ValueError(
                f"Checkpoint provenance mismatch: model={checkpoint.model!r}, "
                f"dpi={checkpoint.dpi}; requested model={args.model!r}, dpi={args.dpi}"
            )
        completed = {
            page_number: embedding
            for page_number, embedding in zip(
                checkpoint.page_numbers, checkpoint.embeddings
            )
        }
        print(f"Resuming {len(completed)} checkpointed pages.")

    for record in records:
        page_number = record["page"]
        if page_number in completed:
            continue
        completed[page_number] = embed_page_images([record["path"]], args.model)[0]
        done = len(completed)
        print(
            f"Embedded page {page_number + 1}/{len(records)} "
            f"({done} total complete)",
            flush=True,
        )
        if done % CHECKPOINT_INTERVAL == 0:
            ordered = [completed[i] for i in sorted(completed)]
            labels = [str(i + 1) for i in sorted(completed)]
            save_index(
                checkpoint_path,
                ordered,
                labels,
                model=args.model,
                dpi=args.dpi,
                page_numbers=sorted(completed),
            )
            print(f"Checkpointed {done} pages to {checkpoint_path}", flush=True)

    page_numbers = sorted(completed)
    save_index(
        output_path,
        [completed[i] for i in page_numbers],
        [str(i + 1) for i in page_numbers],
        model=args.model,
        dpi=args.dpi,
        page_numbers=page_numbers,
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Saved {len(page_numbers)} pages to {output_path}")


if __name__ == "__main__":
    main()
