FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN rm -rf data/chroma_db/experiments/large_800_100 \
    && python scripts/ingest.py \
        --chunk-size 800 \
        --chunk-overlap 100 \
        --persist-dir data/chroma_db/experiments/large_800_100

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
