# 🎥 Project Demo & Reports

## Demo Video

> 📽️ **[https://youtu.be/9g6wdBLimBU](#)**

The demo video (≤ 10 minutes) covers:
- Automatic build & deployment triggered by a GitHub push
- CI/CD pipeline running (Lint & Test + Docker Build & Smoke Test)
- Docker image build for both `train` and `inference` targets
- Application startup and health check
- Live FastAPI endpoint demo (`/health`, `/ready`, `/predict`)
- Example request and response via Swagger UI (`/docs`)

---

## 👥 Team Contributions

| Member | Role | Key Contributions |
|---|---|---|
| Paul Micky D Costa | ML Engineer / Project Setup Lead | Project setup, data preprocessing, baseline model training |
| Devkumar Parikshit Gandhi | DevOps & Automation Engineer | CI/CD pipeline, Dockerization, environment consistency |
| Thai Bao Duong | Serving & Monitoring Engineer | FastAPI inference service, API schema, observability |
| Sofyen Fenich | ML Scientist & Model Validation | Model comparison, evaluation metrics, model explainability |
| Arthur Amuda | Quality & Reproducibility Engineer | Test coverage, MLflow consistency, reproducibility |

---

## 🔧 DevOps Contribution — Devkumar Parikshit Gandhi

### CI/CD Pipeline (`.github/workflows/ci.yml`)
- Automated pipeline triggers on every push and pull request to `main`
- **Job 1 — Lint & Test**: runs pre-commit hooks (Black, Ruff, trailing whitespace) and pytest with `--cov-fail-under=60`
- **Job 2 — Docker Build & Smoke Test**: builds both Docker targets, starts the inference container, and tests `/health`, `/ready`, and `/predict` endpoints automatically

### Multi-Stage Dockerfile
- **Base stage**: Python 3.11-slim with shared dependencies
- **Train target**: runs full preprocessing + training pipeline
- **Inference target**: minimal image serving the FastAPI app on port 8000 with a Docker `HEALTHCHECK`

### Docker Compose (`docker-compose.yml`)
- Easy local development: `docker compose up inference` starts the API instantly
- Training job: `docker compose --profile train up` runs the full pipeline

### Environment Consistency
- All dependencies managed via `pyproject.toml`
- Reproducible builds across local and CI environments

---

## 📊 CI/CD Pipeline Overview

```
Push / Pull Request to main
        │
        ▼
┌─────────────────────┐
│   Lint & Test       │
│  ─────────────────  │
│  pre-commit hooks   │
│  pytest + coverage  │
└────────┬────────────┘
         │ (only if tests pass)
         ▼
┌──────────────────────────┐
│  Docker Build & Smoke    │
│  ──────────────────────  │
│  Build train image       │
│  Build inference image   │
│  Start API container     │
│  Test /health            │
│  Test /ready             │
│  Test /predict           │
└──────────────────────────┘
```

---

## 🗂️ Repository Structure

```
titanic-mlops/
├── .github/workflows/    # CI/CD pipeline (GitHub Actions)
├── src/                  # Core ML code + FastAPI app
├── tests/                # Unit & API tests
├── data/                 # Raw and processed data
├── models/               # Trained model artifacts
├── configs/              # Configuration files
├── scripts/              # Utility scripts
├── docs/                 # Documentation
├── reports/              # This file + demo video link
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # Local development setup
├── pyproject.toml        # Project dependencies
└── README.md             # Main project report
```
