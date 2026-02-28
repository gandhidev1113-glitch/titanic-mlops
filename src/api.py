"""FastAPI inference service for Titanic survival prediction."""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


DEFAULT_LOCAL_MODEL_PATH = "models/baseline_model.pkl"
DEFAULT_FEATURE_COLUMNS_PATH = "models/feature_columns.json"


@dataclass
class ModelBundle:
    """In-memory model and metadata needed for inference."""

    model: object
    source: str
    expected_features: list[str] | None = None


class PredictionRequest(BaseModel):
    """API input schema for a single passenger record."""

    pclass: int = Field(..., ge=1, le=3)
    sex: Literal["male", "female"]
    age: float = Field(..., ge=0, le=100)
    sibsp: int = Field(0, ge=0)
    parch: int = Field(0, ge=0)
    fare: float = Field(..., ge=0)
    embarked: Literal["S", "C", "Q"] = "S"
    title: Literal["Mr", "Miss", "Mrs", "Master", "Rare"] = "Mr"


class PredictionResponse(BaseModel):
    """API output schema."""

    survived: int
    survived_label: str
    probability: float
    model_source: str


_MODEL_BUNDLE: ModelBundle | None = None


def _load_feature_columns(feature_columns_path: str) -> list[str] | None:
    path = Path(feature_columns_path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(col) for col in data]

    return None


def load_model_bundle() -> ModelBundle:
    """Load model from MLflow MODEL_URI or fallback to local pickle artifact."""
    model_uri = os.getenv("MODEL_URI")
    model_path = os.getenv("MODEL_PATH", DEFAULT_LOCAL_MODEL_PATH)
    feature_columns_path = os.getenv("FEATURE_COLUMNS_PATH", DEFAULT_FEATURE_COLUMNS_PATH)
    expected_features = _load_feature_columns(feature_columns_path)

    if model_uri:
        logger.info("Loading inference model from MLflow URI: %s", model_uri)
        model = mlflow.pyfunc.load_model(model_uri)
        return ModelBundle(
            model=model, source=f"mlflow:{model_uri}", expected_features=expected_features
        )

    logger.info("Loading inference model from local artifact: %s", model_path)
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Run training first or set MODEL_URI to an MLflow model."
        )

    with path.open("rb") as f:
        model = pickle.load(f)

    if expected_features is None and hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)

    return ModelBundle(
        model=model, source=f"artifact:{model_path}", expected_features=expected_features
    )


def get_model_bundle() -> ModelBundle:
    """Get cached model bundle, loading lazily on first use."""
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        _MODEL_BUNDLE = load_model_bundle()
    return _MODEL_BUNDLE


def _build_feature_dataframe(payload: PredictionRequest) -> pd.DataFrame:
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    sex_mapping = {"female": 1, "male": 0}

    row = {
        "Pclass": payload.pclass,
        "Sex": sex_mapping[payload.sex],
        "Age": payload.age,
        "SibSp": payload.sibsp,
        "Parch": payload.parch,
        "Fare": payload.fare,
        "Embarked": payload.embarked,
        "Title": title_mapping[payload.title],
    }

    features = pd.DataFrame([row])
    features = pd.get_dummies(features, drop_first=True).fillna(0)
    return features


def _align_features(features: pd.DataFrame, expected_features: list[str] | None) -> pd.DataFrame:
    if not expected_features:
        return features
    return features.reindex(columns=expected_features, fill_value=0)


app = FastAPI(
    title="Titanic Survival Inference API",
    version="1.0.0",
    description="Predict Titanic passenger survival from engineered passenger features.",
)


@app.on_event("startup")
def _warmup_model() -> None:
    try:
        get_model_bundle()
        logger.info("Model warmup completed successfully.")
    except Exception as exc:  # pragma: no cover - exercised by /ready tests
        logger.warning("Model warmup failed: %s", exc)


@app.get("/health")
def health() -> dict:
    """Liveness endpoint."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness endpoint that verifies model availability."""
    try:
        bundle = get_model_bundle()
        return {"status": "ready", "model_source": bundle.source}
    except Exception as exc:
        return {"status": "not_ready", "reason": str(exc)}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Run inference for one passenger and return prediction + probability."""
    try:
        bundle = get_model_bundle()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    features = _build_feature_dataframe(payload)
    aligned = _align_features(features, bundle.expected_features)

    try:
        prediction = int(bundle.model.predict(aligned)[0])
        probability = float(bundle.model.predict_proba(aligned)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    label = "survived" if prediction == 1 else "not_survived"
    return PredictionResponse(
        survived=prediction,
        survived_label=label,
        probability=probability,
        model_source=bundle.source,
    )
