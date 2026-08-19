"""Opt-in visual page retrieval with ColSmol late interaction.

The existing extracted-text retrieval and ranking paths are frozen and remain
untouched. This module deterministically embeds rendered PDF pages, stores
float16 multi-vectors, performs float32 MaxSim scoring through colpali-engine,
and returns page-level results without LLM calls or new labels.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

VISUAL_MODEL = "vidore/colSmol-256M"
DEFAULT_DPI = 150
DEFAULT_INDEX = Path("data/page_index/colsmol_150dpi.npz")

_models = {}
_processors = {}


@dataclass
class VisualIndex:
    """Loaded visual page vectors and their deterministic provenance."""

    embeddings: list
    page_numbers: np.ndarray
    page_labels: list
    model: str
    dpi: int


def get_visual_model(model_name=VISUAL_MODEL):
    """Return one lazily loaded CPU ColSmol model per model name."""
    if model_name not in _models:
        from colpali_engine.models import ColIdefics3

        _models[model_name] = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            attn_implementation="eager",
        ).eval()
    return _models[model_name]


def get_visual_processor(model_name=VISUAL_MODEL):
    """Return one lazily loaded processor per model name."""
    if model_name not in _processors:
        from colpali_engine.models import ColIdefics3Processor

        _processors[model_name] = ColIdefics3Processor.from_pretrained(model_name)
    return _processors[model_name]


def embed_page_images(image_paths, model_name=VISUAL_MODEL):
    """Embed page PNGs and return float16-storage multi-vectors."""
    model = get_visual_model(model_name)
    processor = get_visual_processor(model_name)
    embeddings = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            batch = processor.process_images([image.convert("RGB")]).to("cpu")
        with torch.no_grad():
            embedding = model(**batch)[0].detach().to(torch.float32).cpu().numpy()
        embeddings.append(embedding.astype(np.float16, copy=False))
    return embeddings


def embed_query(question, model_name=VISUAL_MODEL):
    """Embed one text query as a float32 multi-vector."""
    model = get_visual_model(model_name)
    processor = get_visual_processor(model_name)
    batch = processor.process_queries([question]).to("cpu")
    with torch.no_grad():
        return model(**batch)[0].detach().to(torch.float32).cpu()


def save_index(
    path,
    embeddings,
    page_labels,
    model=VISUAL_MODEL,
    dpi=DEFAULT_DPI,
    page_numbers=None,
):
    """Save a padded float16 multi-vector index and provenance metadata."""
    if not embeddings:
        raise ValueError("Cannot save an empty visual index")
    if len(embeddings) != len(page_labels):
        raise ValueError("Embeddings and page labels must have equal length")
    if page_numbers is None:
        page_numbers = list(range(len(embeddings)))
    if len(page_numbers) != len(embeddings):
        raise ValueError("Embeddings and page numbers must have equal length")
    dimensions = {embedding.shape[1] for embedding in embeddings}
    if len(dimensions) != 1:
        raise ValueError("All page embeddings must have the same dimension")
    max_length = max(embedding.shape[0] for embedding in embeddings)
    dimension = dimensions.pop()
    padded = np.zeros((len(embeddings), max_length, dimension), dtype=np.float16)
    lengths = np.empty(len(embeddings), dtype=np.int32)
    for index, embedding in enumerate(embeddings):
        lengths[index] = embedding.shape[0]
        padded[index, : embedding.shape[0]] = embedding
    metadata = {
        "schema_version": 1,
        "model": model,
        "dpi": dpi,
        "dtype": "float16",
        "embedding_dim": dimension,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=padded,
        lengths=lengths,
        page_numbers=np.asarray(page_numbers, dtype=np.int32),
        page_labels=np.asarray(page_labels),
        metadata=np.asarray(json.dumps(metadata)),
    )


def load_index(path=DEFAULT_INDEX):
    """Load and validate one saved visual index."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        embeddings = archive["embeddings"]
        lengths = archive["lengths"]
        page_numbers = archive["page_numbers"]
        page_labels = archive["page_labels"].tolist()
        if metadata.get("dtype") != "float16":
            raise ValueError(f"Unsupported visual index dtype: {metadata.get('dtype')!r}")
        vectors = [
            embeddings[i, : int(length)].astype(np.float32, copy=False)
            for i, length in enumerate(lengths)
        ]
    return VisualIndex(
        embeddings=vectors,
        page_numbers=page_numbers,
        page_labels=page_labels,
        model=metadata["model"],
        dpi=int(metadata["dpi"]),
    )


def score_pages(query_embedding, index):
    """Score all indexed pages with colpali-engine's late interaction."""
    processor = get_visual_processor(index.model)
    pages = [torch.from_numpy(embedding).float() for embedding in index.embeddings]
    scores = processor.score_multi_vector(
        [query_embedding.float()],
        pages,
        device="cpu",
    )
    return scores[0].numpy()


def reference_maxsim(query_embedding, page_embedding):
    """Reference sum-of-max-dot-products implementation for verification."""
    query = query_embedding.float()
    page = torch.from_numpy(page_embedding).float()
    return torch.einsum("qd,pd->qp", query, page).max(dim=1).values.sum().item()


def verify_maxsim_agreement(query_embedding, index, count=2, atol=1e-4):
    """Check colpali-engine scores against the straightforward MaxSim formula."""
    count = min(count, len(index.embeddings))
    processor_scores = score_pages(
        query_embedding,
        VisualIndex(
            embeddings=index.embeddings[:count],
            page_numbers=index.page_numbers[:count],
            page_labels=index.page_labels[:count],
            model=index.model,
            dpi=index.dpi,
        ),
    )
    reference_scores = np.asarray(
        [
            reference_maxsim(query_embedding, embedding)
            for embedding in index.embeddings[:count]
        ]
    )
    if not np.allclose(processor_scores, reference_scores, atol=atol, rtol=1e-5):
        raise ValueError(
            f"MaxSim mismatch: processor={processor_scores.tolist()} "
            f"reference={reference_scores.tolist()}"
        )
    return processor_scores, reference_scores


def retrieve_pages(question, index, top_k=10):
    """Return page records ranked by visual late-interaction score."""
    query_embedding = embed_query(question, index.model)
    scores = score_pages(query_embedding, index)
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [
        {
            "rank": rank,
            "score": float(scores[page_number]),
            "page": int(index.page_numbers[page_number]),
            "page_label": index.page_labels[page_number],
            "model": index.model,
            "dpi": index.dpi,
        }
        for rank, page_number in enumerate(order, start=1)
    ]
