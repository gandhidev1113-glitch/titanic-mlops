# MLOps Final Project - Titanic Survival Prediction

## 📋 Project Description

This project implements an end-to-end ML Ops pipeline for predicting passenger survival on the Titanic. The project follows ML Ops best practices including environment management, code organization, preprocessing, model training, and will eventually include model serving, containerization, and monitoring.

## 🎯 Task Definition

**Problem**: Binary classification task to predict whether a passenger survived the Titanic disaster based on various features such as age, gender, class, fare, etc.

**Objective**: Build a machine learning model that can accurately predict passenger survival and deploy it as a production-ready service following ML Ops principles.

**Evaluation**: Model performance will be evaluated using accuracy, precision, recall, and F1-score metrics.

## 📊 Dataset Source

**Dataset**: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

**Source**: Kaggle Competition Dataset

**Description**: The dataset contains information about 891 passengers in the training set and 418 passengers in the test set. Features include:
- Passenger demographics (Age, Sex)
- Ticket information (Class, Fare, Cabin, Embarked)
- Family information (SibSp, Parch)
- Target variable: Survival (0 = No, 1 = Yes)

**Data Download**: The dataset should be manually downloaded from [Kaggle](https://www.kaggle.com/c/titanic) and placed in `data/raw/` directory. Required files:
- `titanic_train.csv`
- `titanic_test.csv`

> **Note**: An optional download script (`scripts/download_data.py`) is available if you prefer automated download via Kaggle API.

## 👥 Team Member Roles

**Team Members**:
- **PAUL MICKY D COSTA** - ML Engineer / Project Setup Lead
  - Project infrastructure setup and environment management
  - Data pipeline development (preprocessing)
  - Baseline model training and evaluation
  - Code organization and documentation
- **DEVKUMAR PARIKSHIT GANDHI** - DevOps & Automation Engineer
  - CI/CD pipeline setup (GitHub Actions), Dockerization, and environment consistency.
- **Thai Bao DUONG** - Serving & Monitoring Engineer (FastAPI Owner)
   - Build and maintain the FastAPI inference service: POST /predict, GET /health, GET /ready with Pydantic schemas and consistent error handling.
	- Implement model loading interface (artifact path / MODEL_URI) aligned with training outputs and MLflow conventions.
	- Add basic observability: structured logs + simple runtime metrics (latency, request/error counts) suitable for Docker runtime.
	- Write API contract tests (pytest/TestClient) for /predict, /health, /ready to prevent regressions.
	- Backup support for unit testing: contribute additional unit tests when needed (especially around serving-related utilities/interfaces) and assist the Quality Lead in maintaining coverage targets.
- **Sofyen Fenich** - ML Scientist & Model Validation
   - Implement and compare multiple models.
	- Evaluate models using suitable metrics.
	- Model Validation & Data Leakage Prevention
	- Model Explainability & Analysis
	- Keep the best model
- **Arthur Amuda** – Quality & Reproducibility Engineer (Testing & Experiment Governance Lead)
  - Enforce and maintain unit test coverage (≥60%) using pytest an pytest-cov.
  - Configure and maintain CI workflow for automated testing via GitHub Actions.
  - Ensure MLflow experiment logging consistency (parameters, metrics, artifacts).
  - Validate reproducibility of training pipeline across environments.
  - Maintain development standards and checkpoint compliance.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- UV package manager
- Titanic dataset files in `data/raw/` (manually downloaded from [Kaggle](https://www.kaggle.com/c/titanic))

### Setup

1. **Download the dataset**:
   - Go to [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
   - Download `train.csv` and `test.csv`
   - Place them in `data/raw/` as `titanic_train.csv` and `titanic_test.csv`

2. **Install dependencies**:
   ```bash
   uv sync --extra dev
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Run the pipeline**:

   ```bash
   # Using main CLI entry point (recommended)
   python -m src.main preprocess    # Preprocess data
   python -m src.main train         # Train baseline model
   python -m src.main all           # Run full pipeline (preprocess + train)
   ```

   Or using UV:
   ```bash
   uv run python -m src.main preprocess
   uv run python -m src.main train
   ```

   Alternatively, you can run scripts directly:
   ```bash
   # Preprocess data
   python -m src.preprocessing

   # Train baseline model
   python -m src.train
   ```

   > **Optional**: If you want to use automated download via Kaggle API, you can use `python scripts/download_data.py` (requires Kaggle API setup).

## 📁 Project Structure

```
MLOFINAL/
├── data/                  # Data directory
│   ├── raw/              # Raw dataset files (from Kaggle)
│   ├── derived/          # Preprocessed data
│   └── output/           # Generated outputs (predictions, reports)
├── src/                   # Core ML code package
│   ├── __init__.py
│   ├── api.py            # FastAPI inference service
│   ├── main.py           # Main CLI entry point
│   ├── preprocessing.py  # Data preprocessing pipeline
│   ├── train.py          # Model training with MLflow
│   └── utils.py          # Shared utility functions
├── scripts/               # Utility and pipeline scripts
│   ├── __init__.py
│   ├── download_data.py  # Optional: Kaggle API data download (if needed)
│   ├── setup_checkpoint2.bat  # Windows setup script
│   └── setup_checkpoint2.sh   # Linux/Mac setup script
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_preprocessing.py
│   ├── test_train.py
│   └── test_utils.py
├── models/                # Saved models (created during training)
├── docs/                  # Documentation
│   ├── CHECKPOINT2_VERIFICATION.md
│   ├── CONVENTIONAL_COMMITS.md
│   ├── PROFESSOR_FEEDBACK.md
│   └── PROJECT_STRUCTURE.md
├── configs/               # Configuration files (for future use)
├── pyproject.toml         # Project dependencies and config
├── uv.lock                # Locked dependencies
├── Dockerfile             # Docker targets: train / inference
├── .gitignore             # Git ignore rules
├── .pre-commit-config.yaml # Pre-commit hooks config
└── README.md              # This file
```

## 🔄 Workflow

1. **Data Preparation**: Manually download dataset from [Kaggle](https://www.kaggle.com/c/titanic) and place `train.csv` and `test.csv` in `data/raw/` as `titanic_train.csv` and `titanic_test.csv`
2. **Preprocessing**: Run `python -m src.main preprocess` (or `python -m src.preprocessing`) to clean and engineer features
3. **Training**: Execute `python -m src.main train` (or `python -m src.train`) to train the baseline model
4. **Evaluation**: Model metrics are displayed during training

### Quick Start (Full Pipeline)

```bash
python -m src.main all
```

This will run: preprocess → train in sequence (assumes data is already in `data/raw/`).

## 🧪 Testing

### Running Tests

```bash
# Install dev dependencies
uv sync --extra dev

# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py
```

### Test Coverage

The project maintains **79% test coverage** (exceeds 60% requirement). Tests cover:
- Utility functions (`src/utils.py`) - 93% coverage
- Preprocessing pipeline (`src/preprocessing.py`) - 89% coverage
- Training functions (`src/train.py`) - 71% coverage

## 🔧 Pre-commit Hooks

Pre-commit hooks are configured to ensure code quality before commits:

```bash
# Install pre-commit hooks
uv sync --extra dev
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

The hooks check for:
- Code formatting (Black)
- Linting (Ruff)
- Trailing whitespace
- File endings
- YAML/JSON/TOML syntax
- Large files

## 📊 MLflow Experiment Tracking

### Setup

MLflow is integrated for experiment tracking. Experiments are stored locally in `mlruns/` directory.

### Running Training with MLflow

```bash
# Train with default MLflow tracking
python -m src.train

# Or with custom parameters
python -m src.train --experiment-name "titanic_experiments" --n-estimators 300 --max-depth 15
```

### Viewing Experiments

```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

### What's Tracked

- **Parameters**: Model hyperparameters (n_estimators, max_depth, etc.)
- **Metrics**: Accuracy, Precision, Recall, F1-Score (train and validation)
- **Artifacts**:
  - Trained model (registered as `TitanicSurvivalPredictor`)
  - Local model artifact: `models/baseline_model.pkl`
  - Feature schema artifact: `models/feature_columns.json`
  - Feature importance CSV
- **Experiment Naming**: Clear naming convention: `baseline_rf_{n_estimators}trees_{max_depth}depth_{timestamp}`

### Comparing Experiments

Use the MLflow UI to:
1. Compare different runs side-by-side
2. Filter by metrics (e.g., best validation accuracy)
3. View model artifacts and feature importances
4. Track experiment history

## ✅ Checkpoint 2 - Code Quality & Experiment Tracking

**Status**: ✅ Completed

### Deliverables

- ✅ **Pre-commit hooks configured**: `.pre-commit-config.yaml` with Black, Ruff, and standard hooks
- ✅ **Unit tests cover 79%**: Comprehensive test suite in `tests/` directory (exceeds 60% requirement)
- ✅ **Tests runnable locally**: `pytest` command runs all tests
- ✅ **MLflow integrated**:
  - Parameters logged (model hyperparameters, data split info)
  - Metrics logged (accuracy, precision, recall, F1-score)
  - Model artifacts saved and registered
  - Feature importance tracked
- ✅ **Clear experiment naming**: Automatic naming with timestamps and hyperparameters
- ✅ **Meaningful Git commit history**: Structured commits following best practices
- ✅ **README.md updated**: Complete documentation with testing and MLflow sections

### Assessed Skills

- ✅ Software engineering discipline (pre-commit, testing)
- ✅ Testing mindset (79% coverage, comprehensive test cases)
- ✅ Reproducible ML experiments (MLflow tracking, parameter logging)
- ✅ Experiment tracking with MLflow (parameters, metrics, artifacts)

## ✅ Checkpoint 3 - Serving & Containerization

**Status**: ✅ Implemented

### Deliverables

- ✅ **FastAPI application for inference**: `src/api.py`
- ✅ **Clear API schema**:
  - Request model: `PredictionRequest`
  - Response model: `PredictionResponse`
- ✅ **Dockerfile for training and inference**:
  - `docker build --target train ...`
  - `docker build --target inference ...`
- ✅ **Application runnable via Docker** (`uvicorn` in inference target)
- ✅ **Basic API tests**: `tests/test_api.py` with `/health`, `/ready`, `/predict`
- ✅ **Inference model loading from MLflow or artifact**:
  - MLflow via `MODEL_URI`
  - Local fallback via `models/baseline_model.pkl`
- ✅ **README updated** (this section + usage below)

### FastAPI Endpoints

- `GET /health`: liveness check
- `GET /ready`: readiness + model source
- `POST /predict`: single-passenger inference

### API Contract

`POST /predict` request body:

```json
{
  "pclass": 1,
  "sex": "female",
  "age": 29,
  "sibsp": 0,
  "parch": 0,
  "fare": 80.0,
  "embarked": "S",
  "title": "Mrs"
}
```

`POST /predict` response body:

```json
{
  "survived": 1,
  "survived_label": "survived",
  "probability": 0.8,
  "model_source": "artifact:models/baseline_model.pkl"
}
```

### Run Inference API Locally

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open interactive API docs at: `http://localhost:8000/docs`

### Run With Docker

Build and run training image:

```bash
docker build --target train -t titanic-train:latest .
docker run --rm -v "$(pwd)/models:/app/models" titanic-train:latest
```

Build and run inference image:

```bash
docker build --target inference -t titanic-api:latest .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" titanic-api:latest
```

Run inference container against MLflow model URI:

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_URI="models:/TitanicSurvivalPredictor/1" \
  -e FEATURE_COLUMNS_PATH="models/feature_columns.json" \
  -v "$(pwd)/models:/app/models" \
  titanic-api:latest
```

## 📝 Notes

- This project is part of the ML Ops course final project
- **Checkpoint 1**: Project setup, data preprocessing, and baseline model training ✅
- **Checkpoint 2**: Code quality, testing, and MLflow experiment tracking ✅
- **Checkpoint 3**: Model serving (FastAPI), containerization (Docker) ✅
- **Checkpoint 4**: Monitoring, final report, and deployment

## 📚 Additional Documentation

- [Conventional Commits](docs/CONVENTIONAL_COMMITS.md) - Commit message guidelines
- [Project Structure](docs/PROJECT_STRUCTURE.md) - Detailed structure documentation
- [Checkpoint 2 Verification](docs/CHECKPOINT2_VERIFICATION.md) - Verification checklist
- [Professor Feedback](docs/PROFESSOR_FEEDBACK.md) - Feedback and improvements made

## ⚠️ Important Notes

### Repository Naming
- **GitHub Repository**: Should use kebab-case (e.g., `mlo-final` or `titanic-mlops`)
- **Local Folder**: Current folder name is `MLOFINAL` (acceptable for local development)

### File Organization
- ✅ All Python scripts are in `src/` or `scripts/` directories
- ✅ No `.py` files in root directory
- ✅ PDF files excluded from git (see `.gitignore`)

### Commit Convention
- Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format
- See [docs/CONVENTIONAL_COMMITS.md](docs/CONVENTIONAL_COMMITS.md) for guidelines
- Format: `<type>(<scope>): <subject>`
- Example: `feat(train): add MLflow experiment tracking`

### Bonus Points
- Using Titanic dataset (approved by professor)
- Focus on ML Ops best practices and additional features for bonus points
