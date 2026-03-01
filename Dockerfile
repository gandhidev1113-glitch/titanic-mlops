FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
FROM base AS train

COPY configs/ ./configs/
COPY data/ ./data/
RUN mkdir -p models

CMD ["python", "-m", "src.main", "all"]

# ---------------------------------------------------------------------------
FROM base AS inference

COPY configs/ ./configs/
RUN mkdir -p models

ENV MODEL_URI="" \
    FEATURE_COLUMNS_PATH="models/feature_columns.json" \
    HOST="0.0.0.0" \
    PORT="8000"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["sh", "-c", "uvicorn src.api:app --host $HOST --port $PORT"]