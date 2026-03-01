"""Basic API contract tests for the FastAPI inference service."""

from fastapi.testclient import TestClient

from src.api import app, ModelBundle


class DummyModel:
    """Simple deterministic model stub for API testing."""

    def __init__(self) -> None:
        self.feature_names_in_ = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Title",
            "Embarked_Q",
            "Embarked_S",
        ]

    def predict(self, features):
        fare = float(features["Fare"].iloc[0])
        return [1 if fare >= 30 else 0]

    def predict_proba(self, features):
        fare = float(features["Fare"].iloc[0])
        prob = 0.8 if fare >= 30 else 0.2
        return [[1 - prob, prob]]


def _mock_bundle() -> ModelBundle:
    return ModelBundle(
        model=DummyModel(),
        source="artifact:dummy",
        expected_features=DummyModel().feature_names_in_,
    )


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint(monkeypatch):
    monkeypatch.setattr("src.api.get_model_bundle", lambda: _mock_bundle())
    client = TestClient(app)
    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["model_source"] == "artifact:dummy"

def test_ready_endpoint_not_ready(monkeypatch):
    def _raise():
        raise RuntimeError("model missing")

    monkeypatch.setattr("src.api.get_model_bundle", _raise)
    client = TestClient(app)
    response = client.get("/ready")

    assert response.status_code == 503
    assert "Not ready" in response.json()["detail"]


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr("src.api.get_model_bundle", lambda: _mock_bundle())
    client = TestClient(app)

    request_payload = {
        "pclass": 1,
        "sex": "female",
        "age": 29,
        "sibsp": 0,
        "parch": 0,
        "fare": 80.0,
        "embarked": "S",
        "title": "Mrs",
    }
    response = client.post("/predict", json=request_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["survived"] == 1
    assert payload["survived_label"] == "survived"
    assert payload["probability"] > 0.5
    assert payload["model_source"] == "artifact:dummy"


def test_predict_invalid_payload():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "pclass": 4,  # invalid range
            "sex": "unknown",
            "age": -1,
            "fare": -10,
        },
    )

    assert response.status_code == 422
