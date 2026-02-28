# =============================================================================
# Titanic MLOps – Multi-Stage Dockerfile
# Targets:
#   docker build --target train     -t titanic-train:latest .
#   docker build --target inference -t titanic-api:latest   .
# =============================================================================

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

FROM base AS train

RUN uv sync --frozen --no-dev

COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/

RUN mkdir -p models

CMD ["python", "-m", "src.main", "all"]

FROM base AS inference

RUN uv sync --frozen --no-dev

COPY src/ ./src/
COPY configs/ ./configs/

RUN mkdir -p models

ENV MODEL_URI="" \
    FEATURE_COLUMNS_PATH="models/feature_columns.json" \
    HOST="0.0.0.0" \
    PORT="8000"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["sh", "-c", "python -m uvicorn src.api:app --host $HOST --port $PORT"]