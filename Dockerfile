FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY models ./models

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .


FROM base AS train

# Runs preprocessing + training and writes artifacts to /app/models
CMD ["python", "-m", "src.main", "all"]


FROM base AS inference

EXPOSE 8000

ENV MODEL_PATH=models/baseline_model.pkl
ENV FEATURE_COLUMNS_PATH=models/feature_columns.json

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
