# =============================================================================
# Titanic MLOps – Multi-Stage Dockerfile
# Targets:
#   docker build --target train     -t titanic-train:latest .
#   docker build --target inference -t titanic-api:latest   .
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 0 – base: shared Python + UV install
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Keep Python output unbuffered (important for Docker logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install UV (fast Python package manager)
RUN pip install --no-cache-dir uv

# Copy dependency manifests first (layer-cache friendly)
COPY pyproject.toml uv.lock ./

# ---------------------------------------------------------------------------
# Stage 1 – train: installs all deps (incl. dev) and runs training
# ---------------------------------------------------------------------------
FROM base AS train

# Install all dependencies (no dev extras needed for training)
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/

# Model output directory (mount a volume here in production)
RUN mkdir -p models

# Default command: run the full pipeline (preprocess → train)
CMD ["python", "-m", "src.main", "all"]

# ---------------------------------------------------------------------------
# Stage 2 – inference: minimal image, only what the API needs
# ---------------------------------------------------------------------------
FROM base AS inference

# Install only runtime (non-dev) dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Model artifact directory – bind-mount your models/ folder at runtime,
# or bake a model in by uncommenting the next line:
# COPY models/ ./models/

RUN mkdir -p models

# Environment variables (override at runtime as needed)
ENV MODEL_URI="" \
    FEATURE_COLUMNS_PATH="models/feature_columns.json" \
    HOST="0.0.0.0" \
    PORT="8000"

EXPOSE 8000

# Health-check so Docker/orchestrators know when the API is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI application
CMD ["sh", "-c", "python -m uvicorn src.api:app --host $HOST --port $PORT"]
