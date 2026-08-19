"""Deterministic PDF page rendering for the opt-in Pixel RAG path.

The text retrieval artifacts and page identifiers are frozen. This module only
renders the source PDF at a fixed DPI; it never extracts text, calls an LLM,
or changes the existing text pipeline. Existing PNGs are reused so rendering
is idempotent and resumable.
"""

import json
from pathlib import Path

import pymupdf
from PIL import Image

DEFAULT_PDF = Path("data/pdfs/Project_pdf.pdf")
DEFAULT_OUTPUT_DIR = Path("data/page_images")
DEFAULT_DPI = 150
LABEL_POOLS = (
    Path("evaluation/results/hybrid_800_100_candidates_labeled.jsonl"),
    Path("evaluation/results/holdout_v1_hybrid_candidates_labeled.jsonl"),
)


def verify_page_label_mapping(pool_paths=LABEL_POOLS):
    """Fail loudly unless every frozen chunk record uses page + 1 labels."""
    checked = 0
    for pool_path in pool_paths:
        with open(pool_path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                page = record.get("page")
                page_label = record.get("page_label")
                if page_label != str(page + 1):
                    raise ValueError(
                        f"{pool_path}:{line_number} violates page mapping: "
                        f"page={page!r}, page_label={page_label!r}; "
                        "expected page_label == str(page + 1)"
                    )
                checked += 1
    return checked


def render_pdf_pages(
    pdf_path=DEFAULT_PDF,
    output_dir=DEFAULT_OUTPUT_DIR,
    dpi=DEFAULT_DPI,
):
    """Render every PDF page and return deterministic page-image records."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72
    matrix = pymupdf.Matrix(scale, scale)

    records = []
    with pymupdf.open(pdf_path) as document:
        for page_number in range(document.page_count):
            page_path = output_dir / f"page_{page_number + 1:04d}.png"
            page = document.load_page(page_number)
            if not page_path.exists():
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(page_path)
                width, height = pixmap.width, pixmap.height
            else:
                with Image.open(page_path) as image:
                    width, height = image.size
            records.append(
                {
                    "page": page_number,
                    "page_label": str(page_number + 1),
                    "path": str(page_path),
                    "dpi": dpi,
                    "width": width,
                    "height": height,
                    "bytes": page_path.stat().st_size,
                }
            )
    return records
