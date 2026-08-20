FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/opt/huggingface \
    TRANSFORMERS_CACHE=/opt/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/opt/huggingface/sentence_transformers

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$SENTENCE_TRANSFORMERS_HOME" \
    && python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RUN rm -rf data/chroma_db/experiments/large_800_100 \
    && python scripts/ingest.py \
        --chunk-size 800 \
        --chunk-overlap 100 \
        --persist-dir data/chroma_db/experiments/large_800_100

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app "$HF_HOME"
USER appuser

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
