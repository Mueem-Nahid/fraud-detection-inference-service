import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.feature_client import FEATURE_COLUMNS
from app.main import app, init_app


class DummyModelLoader:
    def __init__(self, prediction=1, version="test-version"):
        self.seen_features = None
        self.prediction = prediction
        self.version = version

    def predict(self, features: pd.DataFrame):
        self.seen_features = features.copy()
        return [self.prediction]

    def get_version(self):
        return self.version


class DummyFeatureClient:
    def __init__(self, empty=False):
        self.empty = empty

    def get_features_dataframe(self, entity_id: int) -> pd.DataFrame:
        if self.empty:
            return pd.DataFrame()
        row = {"cc_num": entity_id}
        row.update({name: 1 for name in FEATURE_COLUMNS})
        row.update({"amt": 42.5, "lat": 23.7, "long": 90.4, "merch_lat": 23.8, "merch_long": 90.5, "amt_log": 3.77, "distance_km": 2.4})
        return pd.DataFrame([row])


@pytest.fixture()
def model_loader():
    loader = DummyModelLoader()
    init_app(loader, DummyFeatureClient())
    return loader


@pytest.fixture()
def client(model_loader):
    return TestClient(app)


def test_health_returns_status_and_model_version(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_version": "test-version"}


def test_predict_returns_valid_prediction_and_uses_model_features(client, model_loader):
    response = client.post("/predict", json={"entity_id": 123})

    body = response.json()

    assert response.status_code == 200
    assert body["prediction"] == 1
    assert body["probability"] == 1.0
    assert body["model_version"] == "test-version"
    assert "timestamp" in body
    assert list(model_loader.seen_features.columns) == ["cc_num"] + FEATURE_COLUMNS


def test_predict_rejects_invalid_payload(client):
    response = client.post("/predict", json={"entity_id": "not-an-int"})

    assert response.status_code == 422


def test_predict_returns_404_when_features_are_missing():
    init_app(DummyModelLoader(), DummyFeatureClient(empty=True))
    response = TestClient(app).post("/predict", json={"entity_id": 999})

    assert response.status_code == 404
    assert response.json()["detail"] == "Features not found for entity"


def test_predict_explain_returns_feature_values(client):
    response = client.get("/predict/123?explain=true")

    body = response.json()

    assert response.status_code == 200
    assert body["features"]["amt"] == 42.5
    assert body["features"]["distance_km"] == 2.4


def test_metrics_returns_prometheus_text(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "prediction_requests_total" in response.text
